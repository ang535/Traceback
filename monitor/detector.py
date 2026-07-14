import re
import numpy as np
# get_embedding_model is imported lazily inside compute_goal_similarity rather
# than at module load time — it pulls in sentence-transformers/torch, which is
# only needed when a goal-drift/wrong-file check actually runs. Keeping it out
# of the module-level import lets anything that just imports monitor.detector
# (e.g. agent.agent, or scripts that only exercise other detectors) work
# without that heavy dependency being installed.

MIN_STEPS_FOR_TOKEN_CHECK = 3
ABSOLUTE_TOKEN_CEILING = 4000
LOOP_REPETITION_THRESHOLD = 3

# Empirically set at 0.6 via a 50-scenario labeled benchmark + threshold
# sweep (F1=0.833). Was previously an inline literal default on
# check_goal_drift rather than a named constant — promoted here so
# monitor/scorer.py can import the real validated value instead of keeping
# its own independent (and, until now, inconsistent) copy.
DRIFT_THRESHOLD = 0.6

# Empirically set at 2.2 via tests/measure_token_baseline.py +
# tests/tune_token_threshold_real.py, run against real agent trajectories
# (openai/gpt-oss-20b on Groq) across MULTIPLE trials and MULTIPLE distinct
# tasks (two different one-line bug fixes, run twice each; a structural
# large-output task run five times): the pooled clean ceiling across all
# clean trials was 1.73 (converged consistently across both clean task
# variants), while the weakest of several verbose-task spikes was 2.67.
# 2.2 is the margin-balanced midpoint of that gap — deliberately not the
# smallest value that would still separate the two (which would sit with
# zero margin against clean-run variance the trials didn't happen to
# sample), giving real headroom on both sides instead. Kept in sync with
# monitor/scorer.py's TOKEN_MIN_RATIO, which scales severity off the same value.
TOKEN_MIN_RATIO = 2.2

# A wrong-file mismatch is only flagged if similarity ALSO falls below this
# more lenient bar. This lets genuinely related-but-unnamed files (a
# dependency, a test file, a backup variant) pass, while still catching
# wrong files that are both unnamed AND semantically unrelated.
# Empirically set at 0.65 based on real observed similarity scores: wrong-file
# cases like "buggy_add.py" vs "buggy_multiply.py" scored 0.55-0.65 under
# all-MiniLM-L6-v2, well above what a 0.4-0.5 threshold would catch.
RELATED_FILE_LENIENCY_THRESHOLD = 0.65

# Matches filename-like tokens: word chars / slashes / hyphens, a dot, then
# more word chars. Same pattern used in monitor/validator.py's validate_task,
# kept consistent so both checks recognize filenames the same way.
FILENAME_PATTERN = r"[\w/\-]+\.\w+"


def describe_step(step: dict) -> str:
    """Convert a logged step into a natural-language description for embedding.

    This exists because cosine similarity needs two pieces of comparable text.
    The original goal is a sentence describing intent; a raw tool call
    (e.g. write_file with 50 lines of code as its argument) is not. This
    function bridges that gap by describing the *action*, not dumping the
    raw tool arguments into the embedding.

    Args:
        step: A logged step dict (see monitor.logger.create_step_entry).

    Returns:
        A short natural-language sentence describing what the step did.
    """
    tool = step.get("tool_used", "")
    input_summary = step.get("input_summary", "")

    if tool == "read_file":
        return f"Reading the file {input_summary} to inspect its contents"
    elif tool == "write_file":
        # use only the filepath if input_summary is a dict — file content is
        # implementation detail, not intent, and would add noise to the embedding
        filepath = input_summary.get("filepath", input_summary) if isinstance(input_summary, dict) else input_summary
        return f"Writing changes to the file {filepath}"
    elif tool == "run_code":
        return f"Running the file {input_summary} to test if it works"
    else:
        return str(step.get("output_summary", ""))[:200]


def extract_filenames(text: str) -> set:
    """Extract every filename-shaped token from a piece of text.

    Used to compare which specific files the original task named against
    which specific file the current step actually targeted — a deterministic
    check that catches cases embedding similarity structurally cannot:
    two sentences about "fixing a buggy file" can be nearly identical in
    meaning even when they name two completely different files.

    Args:
        text: Any string — a task description, or a step's input/description.

    Returns:
        A set of filename-shaped substrings found in the text.
    """
    return set(re.findall(FILENAME_PATTERN, str(text)))


def get_step_target_filename(step: dict) -> str | None:
    """Extract the specific file a step actually acted on, if any.

    Args:
        step: A logged step dict.

    Returns:
        The filepath string if one can be determined, otherwise None.
    """
    input_summary = step.get("input_summary")
    if isinstance(input_summary, dict) and "filepath" in input_summary:
        return input_summary["filepath"]
    if isinstance(input_summary, str):
        found = extract_filenames(input_summary)
        if found:
            return next(iter(found))
    return None


def check_wrong_target_file(original_goal: str, current_step: dict, similarity_score: float | None = None,
                             leniency_threshold: float = RELATED_FILE_LENIENCY_THRESHOLD) -> dict | None:
    """Check whether the step's target file is both unnamed AND unrelated to the task.

    This is a combined check: it starts from a deterministic, string-based
    signal (does the step's file match a file actually named in the task)
    but does NOT flag a mismatch on its own anymore. A mismatch only becomes
    an anomaly if the file is ALSO not similar enough to the task semantically
    (similarity_score below leniency_threshold).

    This avoids false positives on legitimate exploration of related-but-
    unnamed files (a dependency, a test file, a backup variant) — those
    typically still score reasonably high similarity even though they weren't
    explicitly named — while still catching files that are both unnamed AND
    clearly unrelated, which is exactly the gap pure embedding similarity
    misses (e.g. buggy_add.py vs buggy_multiply.py score similarly high under
    cosine similarity despite only one being correct).

    Args:
        original_goal: The task description the user originally submitted.
        current_step: The step currently being checked.
        similarity_score: The cosine similarity already computed by
                           check_goal_drift for this same step, reused here
                           rather than recomputed. If None, the function falls
                           back to the original strict (string-only) behavior.
        leniency_threshold: A mismatch is only flagged if similarity falls
                             below this value. More lenient than DRIFT_THRESHOLD
                             on purpose — this check should only fire when both
                             signals agree something is wrong.

    Returns:
        An anomaly dict if the step's file is both an unnamed mismatch AND
        below the leniency threshold, otherwise None.
    """
    task_filenames = extract_filenames(original_goal)
    if not task_filenames:
        return None  # task never named a specific file — nothing to compare against

    step_target = get_step_target_filename(current_step)
    if step_target is None:
        return None  # this step didn't target a specific file (e.g. a reasoning step)

    # normalize by comparing just the basename, so "tasks/buggy_add.py" and
    # "buggy_add.py" are correctly treated as the same file
    task_basenames = {f.split("/")[-1] for f in task_filenames}
    step_basename = step_target.split("/")[-1]

    is_mismatch = step_basename not in task_basenames
    if not is_mismatch:
        return None  # correct file — nothing to flag regardless of similarity

    # a mismatch alone is no longer enough — only flag if the file is ALSO
    # not similar enough to plausibly be reasonable, related exploration
    if similarity_score is not None and similarity_score >= leniency_threshold:
        return None  # wrong file, but similar enough to the task to let it slide

    return {
        "type": "wrong_target_file",
        "step": current_step["step_number"],
        "expected_files": list(task_basenames),
        "actual_file": step_basename,
        "similarity_score": similarity_score,
    }


def compute_goal_similarity(original_goal: str, current_step: dict) -> float:
    """Compute the raw cosine similarity between the task and a step's description.

    Separated out from check_goal_drift so the same similarity score can be
    reused by check_wrong_target_file without recomputing it — embedding a
    sentence is the most expensive part of either check.

    Args:
        original_goal: The task description the user originally submitted.
        current_step: The step currently being checked.

    Returns:
        A float cosine similarity between -1.0 and 1.0.
    """
    from monitor.embeddings import get_embedding_model

    model = get_embedding_model()
    step_description = describe_step(current_step)

    embeddings = model.encode([original_goal, step_description])
    similarity = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
    return similarity


def check_goal_drift(original_goal: str, current_step: dict, trajectory: list,
                      threshold: float = DRIFT_THRESHOLD, similarity_score: float | None = None) -> dict | None:
    """Check whether the current step has semantically drifted from the original goal.

    On the very first step, this still runs, but the result is marked with
    confidence="low" rather than being suppressed entirely — a single data
    point is statistically noisy, so this flag alone should not be enough to
    trigger an automatic rollback (that's enforced downstream, in scorer.py).

    Args:
        original_goal: The task description the user originally submitted.
        current_step: The step currently being checked.
        trajectory: The full active trajectory so far, used to determine
                    whether this is the first step.
        threshold: Cosine similarity below this value is flagged as drift.
        similarity_score: A pre-computed similarity score, if already
                           available, to avoid recomputing it.

    Returns:
        An anomaly dict if drift is detected, otherwise None.
    """
    similarity = similarity_score if similarity_score is not None else compute_goal_similarity(original_goal, current_step)

    is_first_step = len(trajectory) <= 1

    if similarity < threshold:
        return {
            "type": "goal_drift",
            "step": current_step["step_number"],
            "similarity_score": similarity,
            "drifted_target": describe_step(current_step),
            "confidence": "low" if is_first_step else "normal",
        }
    return None


def check_infinite_loop(trajectory: list, threshold: int = LOOP_REPETITION_THRESHOLD) -> dict | None:
    """Check whether the same (tool, input, output) signature has repeated consecutively.

    The signature includes the output, not just the tool and input. A step
    that calls run_code on the same file repeatedly is only a real loop if it
    keeps getting the same result each time — if the output changes between
    calls, the agent is making progress, not looping.

    Args:
        trajectory: The full active trajectory so far.
        threshold: How many consecutive identical signatures count as a loop.

    Returns:
        An anomaly dict if a loop is detected, otherwise None.
    """
    if len(trajectory) < threshold:
        return None

    recent_steps = trajectory[-threshold:]
    signatures = [
        (step["tool_used"], str(step["input_summary"]), str(step["output_summary"]))
        for step in recent_steps
    ]

    if len(set(signatures)) == 1:
        return {
            "type": "infinite_loop",
            "step": trajectory[-1]["step_number"],
            "repeated_signature": signatures[0],
            "repetition_count": threshold,
        }
    return None


def check_token_explosion(trajectory: list, current_step: dict, multiplier: float = TOKEN_MIN_RATIO) -> dict | None:
    """Check whether the current step's token usage spikes relative to recent history.

    Requires at least MIN_STEPS_FOR_TOKEN_CHECK prior steps before this check
    activates — a rolling average computed from 0 or 1 data points is not
    meaningful, so the check is skipped entirely rather than producing a
    misleading result on early steps.

    Args:
        trajectory: The full active trajectory so far, NOT including current_step.
        current_step: The step currently being checked.
        multiplier: How many times the rolling average counts as an explosion.

    Returns:
        An anomaly dict if a spike is detected, otherwise None.
    """
    if len(trajectory) < MIN_STEPS_FOR_TOKEN_CHECK:
        return None

    rolling_avg = sum(s["token_count"] for s in trajectory) / len(trajectory)
    current_tokens = current_step["token_count"]

    if current_tokens > rolling_avg * multiplier:
        return {
            "type": "token_explosion",
            "step": current_step["step_number"],
            "token_count": current_tokens,
            "rolling_average": rolling_avg,
            "ratio": current_tokens / rolling_avg,
            "detection_method": "rolling_average",
        }
    return None


def check_absolute_token_ceiling(current_step: dict, ceiling: int = ABSOLUTE_TOKEN_CEILING) -> dict | None:
    """Check whether a single step's token usage exceeds an absolute hard ceiling.

    Unlike check_token_explosion, this requires no history and is active from
    step 1. It exists specifically to cover the blind spot in the first few
    steps of every trajectory, where the relative/statistical checks have not
    yet accumulated enough data to be reliable.

    Args:
        current_step: The step currently being checked.
        ceiling: The absolute token count above which a step is flagged.

    Returns:
        An anomaly dict if the ceiling is exceeded, otherwise None.
    """
    if current_step["token_count"] > ceiling:
        return {
            "type": "token_explosion",
            "step": current_step["step_number"],
            "token_count": current_step["token_count"],
            "ceiling": ceiling,
            "detection_method": "absolute_ceiling",
        }
    return None


def run_all_detectors(original_goal: str, trajectory: list, current_step: dict) -> list:
    """Run every detector against the current step and return all triggered anomalies.

    Args:
        original_goal: The task description the user originally submitted.
        trajectory: The active trajectory BEFORE current_step was added.
        current_step: The step currently being checked.

    Returns:
        A list of anomaly dicts. Empty if nothing was flagged.
    """
    anomalies = []

    # compute similarity once, shared between check_goal_drift and
    # check_wrong_target_file — avoids embedding the same sentence twice
    similarity_score = compute_goal_similarity(original_goal, current_step)

    drift = check_goal_drift(original_goal, current_step, trajectory + [current_step], similarity_score=similarity_score)
    if drift:
        anomalies.append(drift)

    wrong_file = check_wrong_target_file(original_goal, current_step, similarity_score=similarity_score)
    if wrong_file:
        anomalies.append(wrong_file)

    loop = check_infinite_loop(trajectory + [current_step])
    if loop:
        anomalies.append(loop)

    token_spike = check_token_explosion(trajectory, current_step)
    if token_spike:
        anomalies.append(token_spike)

    token_ceiling = check_absolute_token_ceiling(current_step)
    if token_ceiling:
        anomalies.append(token_ceiling)

    return anomalies 