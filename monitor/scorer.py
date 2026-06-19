BASE_SEVERITY = {
    "goal_drift": 0.6,
    "infinite_loop": 0.5,
    "token_explosion": 0.55,
}

LOW_CONFIDENCE_PENALTY = 0.5  # how much a "low confidence" flag gets discounted

DIMINISHING_RETURNS_RATE = 0.3  # each additional anomaly closes 30% of the remaining gap to 1.0


def get_anomaly_base_score(anomaly: dict) -> float:
    """Look up the base severity score for a single anomaly, adjusted for confidence.

    Args:
        anomaly: A single anomaly dict, as produced by monitor.detector functions.

    Returns:
        A float severity score for this one anomaly, before combining with others.
    """
    base = BASE_SEVERITY.get(anomaly["type"], 0.5)

    if anomaly.get("confidence") == "low":
        base = base * LOW_CONFIDENCE_PENALTY

    return base


def calculate_severity(anomalies: list) -> float:
    """Combine a list of simultaneous anomalies into a single severity score.

    If no anomalies fired, severity is 0.0. If exactly one fired, severity is
    just its base score. If multiple fired at once, each additional anomaly
    closes DIMINISHING_RETURNS_RATE of the remaining distance to 1.0, rather
    than simply summing scores (which could exceed 1.0 and lose meaning).

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