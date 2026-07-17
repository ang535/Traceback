"""
Finds the optimal SEVERITY_FULL_RESTART_THRESHOLD using the real, two-signal
gate from monitor.rollback.find_rollback_point — combined severity from
monitor.scorer.calculate_severity() AND distinct anomaly type count, exactly
as the actual code evaluates it.

Same offline methodology as tune_severity_threshold.py: no live agent runs or
LLM calls needed. Sweep severity-threshold candidates (with the type-count gate
held fixed at MIN_ANOMALY_TYPES_FOR_FULL_RESTART, since that's a separate,
already-decided design choice — see monitor/rollback.py), evaluate
precision/recall/F1 against tests/fixtures/full_restart_scenarios.py, report
the best.

Read the caveat at the top of full_restart_scenarios.py before trusting this
the way TOKEN_MIN_RATIO's tuning was trusted: full_restart has never fired in
a real run, so these labels are reasoned judgment calls, not measurements
from observed incidents.

HISTORY: the first version of this script swept severity alone (no type-count
gate) and topped out at F1=0.706, because a single maxed-out anomaly and a
genuine multi-signal compound failure can both compute to severity=1.0 —
indistinguishable by severity alone. That finding led to adding the
distinct-anomaly-type-count condition to find_rollback_point itself (not just
to this test). This version evaluates the real two-signal gate.

Run with:
    python3 -m tests.tune_full_restart_threshold
"""

from monitor.scorer import calculate_severity
from monitor.rollback import MIN_ANOMALY_TYPES_FOR_FULL_RESTART
from tests.fixtures.full_restart_scenarios import FULL_RESTART_SCENARIOS

CANDIDATES = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def evaluate_threshold(rows: list, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    wrong = []

    for severity, type_count, should_full_restart, description in rows:
        predicted = severity >= threshold and type_count >= MIN_ANOMALY_TYPES_FOR_FULL_RESTART

        if predicted and should_full_restart:
            tp += 1
        elif predicted and not should_full_restart:
            fp += 1
            wrong.append(("FP", severity, type_count, description))
        elif not predicted and not should_full_restart:
            tn += 1
        else:
            fn += 1
            wrong.append(("FN", severity, type_count, description))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "wrong": wrong,
    }


if __name__ == "__main__":
    print(f"Computing real severity + type-count for {len(FULL_RESTART_SCENARIOS)} labeled scenarios "
          f"via the real monitor.scorer.calculate_severity() "
          f"(type-count gate fixed at >= {MIN_ANOMALY_TYPES_FOR_FULL_RESTART})...\n")

    rows = []
    for scenario in FULL_RESTART_SCENARIOS:
        severity = calculate_severity(scenario["anomalies"])
        type_count = len({a["type"] for a in scenario["anomalies"]})
        rows.append((severity, type_count, scenario["should_full_restart"], scenario["description"]))
        label = "should full_restart" if scenario["should_full_restart"] else "should NOT full_restart"
        print(f"  severity={severity:<8} types={type_count}  ({label:<23}) — {scenario['description'][:75]}")

    print()
    results = [evaluate_threshold(rows, t) for t in CANDIDATES]
    best = max(results, key=lambda r: r["f1"])

    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}"
          f"{'TP':<6}{'FP':<6}{'TN':<6}{'FN':<6}")
    print("-" * 70)
    for r in results:
        marker = " <-- best F1" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:<12}{r['precision']:<12}{r['recall']:<12}"
              f"{r['f1']:<10}{r['tp']:<6}{r['fp']:<6}{r['tn']:<6}{r['fn']:<6}{marker}")

    print(f"\nBest threshold: {best['threshold']}  "
          f"(F1={best['f1']}, Precision={best['precision']}, Recall={best['recall']})")
    print(f"Current value in monitor/rollback.py: SEVERITY_FULL_RESTART_THRESHOLD = 0.85")

    if best["wrong"]:
        print(f"\nMisclassified scenarios at threshold={best['threshold']}:")
        for outcome, severity, type_count, description in best["wrong"]:
            print(f"  [{outcome}] severity={severity} types={type_count}  {description[:80]}")
    else:
        print(f"\nPerfect classification at threshold={best['threshold']} — no misclassified scenarios.")
