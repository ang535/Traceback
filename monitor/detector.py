import numpy as np
from monitor.embeddings import get_embedding_model

MIN_STEPS_FOR_TOKEN_CHECK = 3
ABSOLUTE_TOKEN_CEILING = 4000
LOOP_REPETITION_THRESHOLD = 3


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


def check_goal_drift(original_goal: str, current_step: dict, trajectory: list, threshold: float = 0.4) -> dict | None:
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

    Returns:
        An anomaly dict if drift is detected, otherwise None.
    """
    model = get_embedding_model()
    step_description = describe_step(current_step)

    embeddings = model.encode([original_goal, step_description])
    similarity = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )

    is_first_step = len(trajectory) <= 1

    if similarity < threshold:
        return {
            "type": "goal_drift",
            "step": current_step["step_number"],
            "similarity_score": similarity,
            "drifted_target": step_description,
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


def check_token_explosion(trajectory: list, current_step: dict, multiplier: float = 3.0) -> dict | None:
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

    drift = check_goal_drift(original_goal, current_step, trajectory + [current_step])
    if drift:
        anomalies.append(drift)

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