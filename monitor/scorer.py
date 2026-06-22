LOW_CONFIDENCE_PENALTY = 0.5  # how much a "low confidence" flag gets discounted
DIMINISHING_RETURNS_RATE = 0.3  # each additional anomaly closes 30% of the remaining gap to 1.0

# Tuning constants for each detector's magnitude-to-severity mapping.
# These define the scale, not the per-anomaly outcome — the actual score is
# always derived from the real numbers each detector measured.
DRIFT_THRESHOLD = 0.4

LOOP_MIN_THRESHOLD = 3   # repetitions needed to trigger the anomaly at all
LOOP_SEVERITY_FLOOR = 0.5  # severity at exactly the minimum threshold
LOOP_CAP_REPETITIONS = 6   # repetitions at which severity saturates to 1.0

TOKEN_MIN_RATIO = 3.0     # ratio needed to trigger the anomaly at all
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
        # absolute hard ceiling) but have no relative magnitude to scale by
        return 0.6

    ratio = anomaly["ratio"]
    span = cap_ratio - min_ratio
    progress = (ratio - min_ratio) / span if span > 0 else 1.0
    severity = floor + (1.0 - floor) * progress
    return max(floor, min(severity, 1.0))


SCORERS = {
    "goal_drift": score_goal_drift,
    "infinite_loop": score_infinite_loop,
    "token_explosion": score_token_explosion,
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