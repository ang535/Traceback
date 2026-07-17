from monitor.detector import DRIFT_THRESHOLD, LOOP_REPETITION_THRESHOLD  # noqa: F401 — re-exported; see notes below

# Checked via tests/tune_severity_formula_params.py: a sensitivity analysis
# that reuses the labeled scenarios in tests/fixtures/severity_scenarios.py
# (already validated against the real ROLLBACK_SEVERITY_THRESHOLD=0.45) as an
# end-to-end check on the FINAL should_rollback decision, not just the raw
# severity number. Unlike TOKEN_MIN_RATIO or ROLLBACK_SEVERITY_THRESHOLD,
# these constants aren't independent thresholds with a clean labeled ground
# truth of their own — they're internal shape parameters of
# calculate_severity(), so this asks a different question: how much does
# each one actually matter, given everything else at its real value?
#
# LOW_CONFIDENCE_PENALTY was the one real finding. At the old value (0.5), a
# single low-confidence goal_drift signal (similarity=0.0, confidence="low")
# computes severity exactly 0.5 — just over ROLLBACK_SEVERITY_THRESHOLD
# (0.45), incorrectly triggering a rollback despite the whole point of the
# penalty being to NOT overreact to one noisy signal alone. Sweeping
# candidates [0.0, 0.25, 0.5, 0.75, 1.0] showed F1 improves from 0.929 to
# 0.963 at 0.0 or 0.25, and stays there — both remove that false positive.
# 0.25 was chosen over the more extreme 0.0: 0.0 fully zeroes out any
# low-confidence signal even when corroborating with others, discarding real
# information; 0.25 fixes the actual false positive (a low-confidence signal
# ALONE) while still letting a low-confidence signal contribute a small
# amount toward corroborating other anomalies, matching the project's stated
# philosophy that "corroborating signals are stronger evidence than any one
# of them alone."
LOW_CONFIDENCE_PENALTY = 0.25  # how much a "low confidence" flag gets discounted

# DIMINISHING_RETURNS_RATE: swept [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] — F1
# stayed at 0.929 (the ROLLBACK_SEVERITY_THRESHOLD-validated ceiling for this
# scenario set) across the ENTIRE range. 0.3 kept as a reasonable mid-range
# choice — the system is provably insensitive to this constant.
DIMINISHING_RETURNS_RATE = 0.3  # each additional anomaly closes 30% of the remaining gap to 1.0

# Tuning constants for each detector's magnitude-to-severity mapping.
# These define the scale, not the per-anomaly outcome — the actual score is
# always derived from the real numbers each detector measured.
#
# LOOP_SEVERITY_FLOOR: swept [0.3, 0.4, 0.5, 0.6, 0.7] — 0.5 and above all
# score the best F1 (0.929); below 0.5 an isolated bare-minimum infinite_loop
# stops clearing ROLLBACK_SEVERITY_THRESHOLD, missing the exact case
# ROLLBACK_SEVERITY_THRESHOLD's own tuning was built around. 0.5 kept — the
# smallest value in the robust range.
#
# LOOP_CAP_REPETITIONS: swept [4, 5, 6, 8, 10] — F1 constant at 0.929 across
# the entire range. Kept at 6 as a reasonable mid-range choice.
#
# TOKEN_SEVERITY_FLOOR: swept [0.2, 0.3, 0.4, 0.5, 0.6] — 0.4 is the ONLY
# value achieving the best F1 (0.929); both directions make it worse (0.2-0.3
# miss a real case, 0.5-0.6 add false positives). Confirmed correct, not just
# unexamined.
#
# TOKEN_CAP_RATIO: swept [6, 8, 10, 15, 20] — F1 constant at 0.929 across the
# entire range. Kept at 10.0 as a reasonable mid-range choice.
#
# DRIFT_THRESHOLD used to be redefined here as 0.4, independently of
# monitor.detector's real (empirically validated) cutoff of 0.6. That meant
# any goal_drift anomaly with similarity between 0.4 and 0.6 — genuinely
# flagged, since 0.6 is the actual detection cutoff — computed a NEGATIVE
# severity via (0.4 - similarity) / 0.4, clamped to 0.0. A real band of
# correctly-detected drift was silently scoring zero severity. Now imported
# directly from monitor.detector so there's one source of truth.

# LOOP_MIN_THRESHOLD used to be redefined here as its own literal 3,
# independently of monitor.detector's real (empirically validated)
# LOOP_REPETITION_THRESHOLD. Both happened to equal 3, so nothing was
# actively broken — but it was the exact same bug shape as the DRIFT_THRESHOLD
# issue above: two disconnected copies of the same real-world number, with
# nothing keeping them in sync if one is ever changed. Now imported directly.
LOOP_MIN_THRESHOLD = LOOP_REPETITION_THRESHOLD  # repetitions needed to trigger the anomaly at all
LOOP_SEVERITY_FLOOR = 0.5  # severity at exactly the minimum threshold
LOOP_CAP_REPETITIONS = 6   # repetitions at which severity saturates to 1.0

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
# sample), giving real headroom on both sides instead.
TOKEN_MIN_RATIO = 2.2    # ratio needed to trigger the anomaly at all
TOKEN_SEVERITY_FLOOR = 0.4  # severity at exactly the minimum ratio
TOKEN_CAP_RATIO = 10.0    # ratio at which severity saturates to 1.0


def score_goal_drift(anomaly: dict, threshold: float = DRIFT_THRESHOLD) -> float:
    """Score a goal_drift anomaly based on how far below threshold the similarity fell.

    A similarity just under the threshold is barely a drift; a similarity
    near 0 (completely unrelated) is a severe drift. This scales linearly
    between those two points.

    Args:
        anomaly: A goal_drift anomaly dict, expected to have "similarity_score".
        threshold: The similarity threshold below which drift is flagged.

    Returns:
        A float between 0.0 and 1.0.
    """
    similarity = anomaly["similarity_score"]
    severity = (threshold - similarity) / threshold
    return max(0.0, min(severity, 1.0))


def score_infinite_loop(anomaly: dict, min_threshold: int = LOOP_MIN_THRESHOLD,
                         cap_repetitions: int = LOOP_CAP_REPETITIONS,
                         floor: float = LOOP_SEVERITY_FLOOR) -> float:
    """Score an infinite_loop anomaly based on how many repetitions occurred.

    Hitting the minimum threshold at all is already a real problem (hence the
    floor), and severity rises toward 1.0 as repetitions climb past that,
    saturating once cap_repetitions is reached.

    Args:
        anomaly: An infinite_loop anomaly dict, expected to have "repetition_count".
        min_threshold: The minimum repetitions needed to trigger this anomaly.
        cap_repetitions: Repetitions at which severity saturates to 1.0.
        floor: The severity assigned at exactly min_threshold repetitions.

    Returns:
        A float between 0.0 and 1.0.
    """
    repetitions = anomaly["repetition_count"]
    span = cap_repetitions - min_threshold
    progress = (repetitions - min_threshold) / span if span > 0 else 1.0
    severity = floor + (1.0 - floor) * progress
    return max(floor, min(severity, 1.0))


def score_token_explosion(anomaly: dict, min_ratio: float = TOKEN_MIN_RATIO,
                           cap_ratio: float = TOKEN_CAP_RATIO,
                           floor: float = TOKEN_SEVERITY_FLOOR) -> float:
    """Score a token_explosion anomaly based on how far the ratio exceeds the trigger point.

    Works for both detection methods (rolling_average and absolute_ceiling) as
    long as the anomaly includes a "ratio" or can derive one; falls back to a
    flat moderate score if no ratio is available (e.g. absolute ceiling hits
    where no rolling average exists yet).

    Args:
        anomaly: A token_explosion anomaly dict.
        min_ratio: The ratio needed to trigger this anomaly via rolling average.
        cap_ratio: Ratio at which severity saturates to 1.0.
        floor: The severity assigned at exactly min_ratio.

    Returns:
        A float between 0.0 and 1.0.
    """
    if anomaly.get("detection_method") == "absolute_ceiling" or "ratio" not in anomaly:
        # no rolling average existed yet — we know it's bad (it broke an
        # absolute hard ceiling) but have no relative magnitude to scale by.
        # Checked via tests/tune_severity_formula_params.py: swept
        # [0.4, 0.5, 0.6, 0.7, 0.8] against the labeled scenarios — F1 stays
        # at the best value (0.929) for 0.5-0.8, only 0.4 is worse. 0.6 kept.
        return 0.6

    ratio = anomaly["ratio"]
    span = cap_ratio - min_ratio
    progress = (ratio - min_ratio) / span if span > 0 else 1.0
    severity = floor + (1.0 - floor) * progress
    return max(floor, min(severity, 1.0))


def score_wrong_target_file(anomaly: dict) -> float:
    """Score a wrong_target_file anomaly.

    Unlike the other checks, this is a deterministic, binary signal — the
    step's target file either matches a file named in the task, or it
    doesn't. There's no "how wrong" gradient to scale by (a wrong file is
    a wrong file), so this returns a flat, high severity rather than a
    magnitude-scaled one.

    This is intentionally high — higher than goal_drift's typical scores —
    because this check only ever fires on a confident, exact mismatch with
    no ambiguity, unlike embedding similarity which can be uncertain.

    Checked via tests/tune_severity_formula_params.py: swept
    [0.5, 0.6, 0.7, 0.8, 0.9] against the labeled scenarios — F1 stays at
    the best value (0.929) across the entire range. 0.7 kept as a
    reasonable mid-range choice.

    Args:
        anomaly: A wrong_target_file anomaly dict.

    Returns:
        A fixed float severity (0.7).
    """
    return 0.7


SCORERS = {
    "goal_drift": score_goal_drift,
    "infinite_loop": score_infinite_loop,
    "token_explosion": score_token_explosion,
    "wrong_target_file": score_wrong_target_file,
}


def get_anomaly_base_score(anomaly: dict) -> float:
    """Calculate the severity score for a single anomaly from its actual measured values.

    Dispatches to the correct scoring function based on anomaly type, then
    applies the confidence penalty if the anomaly was flagged as low-confidence
    (e.g. goal drift detected on the very first step, with no prior history).

    Args:
        anomaly: A single anomaly dict, as produced by monitor.detector functions.

    Returns:
        A float severity score for this one anomaly, before combining with others.
    """
    scorer_fn = SCORERS.get(anomaly["type"])
    base = scorer_fn(anomaly) if scorer_fn else 0.5

    if anomaly.get("confidence") == "low":
        base = base * LOW_CONFIDENCE_PENALTY

    return base


def calculate_severity(anomalies: list) -> float:
    """Combine a list of simultaneous anomalies into a single severity score.

    If no anomalies fired, severity is 0.0. If exactly one fired, severity is
    just its measured score. If multiple fired at once, each additional
    anomaly closes DIMINISHING_RETURNS_RATE of the remaining distance to 1.0,
    rather than simply summing scores (which could exceed 1.0 and lose meaning).

    Args:
        anomalies: A list of anomaly dicts, as returned by run_all_detectors.

    Returns:
        A float between 0.0 and 1.0 representing combined severity.
    """
    if not anomalies:
        return 0.0

    scores = sorted((get_anomaly_base_score(a) for a in anomalies), reverse=True)

    severity = scores[0]
    for score in scores[1:]:
        remaining_gap = 1.0 - severity
        severity += remaining_gap * DIMINISHING_RETURNS_RATE * score

    return round(min(severity, 1.0), 4)