"""
Sensitivity analysis for the severity-formula "shape" constants in
monitor/scorer.py: LOW_CONFIDENCE_PENALTY, DIMINISHING_RETURNS_RATE,
LOOP_SEVERITY_FLOOR, LOOP_CAP_REPETITIONS, TOKEN_SEVERITY_FLOOR,
TOKEN_CAP_RATIO, the flat 0.7 in score_wrong_target_file, and the flat 0.6
absolute_ceiling fallback in score_token_explosion.

Why this needs a different methodology than TOKEN_MIN_RATIO or
ROLLBACK_SEVERITY_THRESHOLD: those are independent thresholds on a single raw
signal (a ratio, a composite score) with a clean should/shouldn't-fire
question to label. These eight are internal shape parameters of
calculate_severity() itself — several steps removed from any labeled
outcome, and none of them individually has a clean "right" value the way a
detection cutoff does.

What this script does instead: reuses the existing labeled scenario set
(tests/fixtures/severity_scenarios.py, already validated against the real
ROLLBACK_SEVERITY_THRESHOLD=0.45) as an END-TO-END check. For each
parameter, holds ROLLBACK_SEVERITY_THRESHOLD and every OTHER parameter at
its real production default, sweeps ONLY that one parameter across a range
of plausible values, and measures how much the FINAL should_rollback
decision (not just the raw severity number) actually changes.

This answers a more useful question than "what's the best value": how much
does this constant actually matter? If F1 stays at the current best (0.929)
across a wide range, the system is robust to it and the current value is a
reasonable, low-stakes choice that doesn't need further tuning. If F1 is
sensitive to it, that constant deserves the same scenario-labeling rigor the
real thresholds got, not just a sweep like this one.

The scoring logic below mirrors monitor/scorer.py exactly, with each swept
constant exposed as a parameter instead of hardcoded — see the docstring on
each function for which line(s) of the real file it mirrors.

Run with:
    python3 -m tests.tune_severity_formula_params
"""

from monitor.scorer import (
    LOOP_MIN_THRESHOLD, TOKEN_MIN_RATIO,
    score_goal_drift, score_infinite_loop,
    LOW_CONFIDENCE_PENALTY, DIMINISHING_RETURNS_RATE,
    LOOP_SEVERITY_FLOOR, LOOP_CAP_REPETITIONS,
    TOKEN_SEVERITY_FLOOR, TOKEN_CAP_RATIO,
)
from tests.fixtures.severity_scenarios import SEVERITY_SCENARIOS

ROLLBACK_SEVERITY_THRESHOLD = 0.45  # the real, already-validated gate — held fixed throughout
TOKEN_ABSOLUTE_CEILING_FALLBACK_DEFAULT = 0.6  # hardcoded in the real score_token_explosion
WRONG_TARGET_FLAT_DEFAULT = 0.7  # hardcoded in the real score_wrong_target_file


def score_token_explosion_param(anomaly, min_ratio=TOKEN_MIN_RATIO, cap_ratio=TOKEN_CAP_RATIO,
                                 floor=TOKEN_SEVERITY_FLOOR,
                                 absolute_ceiling_fallback=TOKEN_ABSOLUTE_CEILING_FALLBACK_DEFAULT):
    """Mirrors monitor.scorer.score_token_explosion. The real function's
    min_ratio/cap_ratio/floor are already swappable parameters there; this
    additionally exposes the absolute_ceiling fallback (a bare `return 0.6`
    in production) as a parameter, since that's one of the constants
    under test here."""
    if anomaly.get("detection_method") == "absolute_ceiling" or "ratio" not in anomaly:
        return absolute_ceiling_fallback
    ratio = anomaly["ratio"]
    span = cap_ratio - min_ratio
    progress = (ratio - min_ratio) / span if span > 0 else 1.0
    severity = floor + (1.0 - floor) * progress
    return max(floor, min(severity, 1.0))


def score_wrong_target_file_param(anomaly, flat_severity=WRONG_TARGET_FLAT_DEFAULT):
    """Mirrors monitor.scorer.score_wrong_target_file, with the flat
    severity (a bare `return 0.7` in production) exposed as a parameter."""
    return flat_severity


def get_anomaly_base_score_param(anomaly, low_confidence_penalty=LOW_CONFIDENCE_PENALTY,
                                  loop_floor=LOOP_SEVERITY_FLOOR, loop_cap=LOOP_CAP_REPETITIONS,
                                  token_floor=TOKEN_SEVERITY_FLOOR, token_cap=TOKEN_CAP_RATIO,
                                  token_ceiling_fallback=TOKEN_ABSOLUTE_CEILING_FALLBACK_DEFAULT,
                                  wrong_target_flat=WRONG_TARGET_FLAT_DEFAULT):
    """Mirrors monitor.scorer.get_anomaly_base_score's dispatch + confidence
    penalty, threading the swept parameters through to whichever scorer
    actually applies to this anomaly's type."""
    anomaly_type = anomaly["type"]

    if anomaly_type == "goal_drift":
        base = score_goal_drift(anomaly)
    elif anomaly_type == "infinite_loop":
        base = score_infinite_loop(anomaly, min_threshold=LOOP_MIN_THRESHOLD,
                                    cap_repetitions=loop_cap, floor=loop_floor)
    elif anomaly_type == "token_explosion":
        base = score_token_explosion_param(anomaly, min_ratio=TOKEN_MIN_RATIO,
                                            cap_ratio=token_cap, floor=token_floor,
                                            absolute_ceiling_fallback=token_ceiling_fallback)
    elif anomaly_type == "wrong_target_file":
        base = score_wrong_target_file_param(anomaly, flat_severity=wrong_target_flat)
    else:
        base = 0.5

    if anomaly.get("confidence") == "low":
        base = base * low_confidence_penalty

    return base


def calculate_severity_param(anomalies, diminishing_returns_rate=DIMINISHING_RETURNS_RATE, **kwargs):
    """Mirrors monitor.scorer.calculate_severity's combination formula,
    with diminishing_returns_rate exposed as a parameter and every other
    swept parameter forwarded through **kwargs to get_anomaly_base_score_param."""
    if not anomalies:
        return 0.0

    scores = sorted((get_anomaly_base_score_param(a, **kwargs) for a in anomalies), reverse=True)

    severity = scores[0]
    for score in scores[1:]:
        remaining_gap = 1.0 - severity
        severity += remaining_gap * diminishing_returns_rate * score

    return round(min(severity, 1.0), 4)


def evaluate(param_kwarg: str, candidates: list) -> list:
    results = []
    for value in candidates:
        kwargs = {param_kwarg: value}
        tp = fp = tn = fn = 0
        for scenario in SEVERITY_SCENARIOS:
            severity = calculate_severity_param(scenario["anomalies"], **kwargs)
            predicted = severity >= ROLLBACK_SEVERITY_THRESHOLD
            actual = scenario["should_rollback"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append({
            "value": value, "f1": round(f1, 3), "precision": round(precision, 3),
            "recall": round(recall, 3), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return results


SWEEPS = [
    ("LOW_CONFIDENCE_PENALTY", "low_confidence_penalty", LOW_CONFIDENCE_PENALTY,
     [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("DIMINISHING_RETURNS_RATE", "diminishing_returns_rate", DIMINISHING_RETURNS_RATE,
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]),
    ("LOOP_SEVERITY_FLOOR", "loop_floor", LOOP_SEVERITY_FLOOR,
     [0.3, 0.4, 0.5, 0.6, 0.7]),
    ("LOOP_CAP_REPETITIONS", "loop_cap", LOOP_CAP_REPETITIONS,
     [4, 5, 6, 8, 10]),
    ("TOKEN_SEVERITY_FLOOR", "token_floor", TOKEN_SEVERITY_FLOOR,
     [0.2, 0.3, 0.4, 0.5, 0.6]),
    ("TOKEN_CAP_RATIO", "token_cap", TOKEN_CAP_RATIO,
     [6, 8, 10, 15, 20]),
    ("score_token_explosion's absolute_ceiling fallback", "token_ceiling_fallback",
     TOKEN_ABSOLUTE_CEILING_FALLBACK_DEFAULT, [0.4, 0.5, 0.6, 0.7, 0.8]),
    ("score_wrong_target_file's flat severity", "wrong_target_flat",
     WRONG_TARGET_FLAT_DEFAULT, [0.5, 0.6, 0.7, 0.8, 0.9]),
]


if __name__ == "__main__":
    print(f"Sensitivity analysis against {len(SEVERITY_SCENARIOS)} labeled scenarios, "
          f"ROLLBACK_SEVERITY_THRESHOLD fixed at {ROLLBACK_SEVERITY_THRESHOLD} "
          f"(the real, already-validated value).\n")

    for name, kwarg, current_default, candidates in SWEEPS:
        results = evaluate(kwarg, candidates)
        best_f1 = max(r["f1"] for r in results)
        robust_range = [r["value"] for r in results if r["f1"] == best_f1]

        print(f"=== {name} (current default: {current_default}) ===")
        print(f"{'Value':<10}{'F1':<8}{'Precision':<12}{'Recall':<10}{'TP':<5}{'FP':<5}{'TN':<5}{'FN':<5}")
        for r in results:
            marker = " <-- current default" if r["value"] == current_default else ""
            marker += " <-- best F1" if r["f1"] == best_f1 and r["value"] != current_default else ""
            print(f"{str(r['value']):<10}{r['f1']:<8}{r['precision']:<12}{r['recall']:<10}"
                  f"{r['tp']:<5}{r['fp']:<5}{r['tn']:<5}{r['fn']:<5}{marker}")

        current_result = next((r for r in results if r["value"] == current_default), None)
        if current_result and current_result["f1"] == best_f1:
            print(f"-> Current default {current_default} is within the best-F1 range {robust_range}. "
                  f"System is robust to this constant across the tested range.")
        elif current_result:
            print(f"-> Current default {current_default} scores F1={current_result['f1']}, "
                  f"below the best F1={best_f1} achieved at {robust_range}.")
        print()
