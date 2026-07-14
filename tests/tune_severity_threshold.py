"""
Finds the optimal ROLLBACK_SEVERITY_THRESHOLD using real severity scores
computed the SAME WAY as agent.py's rollback gate — via the actual, unmodified
monitor.scorer.calculate_severity().

Unlike drift/loop/token tuning, this needs no live agent runs or LLM calls:
calculate_severity() is pure Python, so real severity numbers can be computed
directly from labeled anomaly-dict scenarios (tests/fixtures/severity_scenarios.py).
This also means today's Groq quota isn't a constraint on this run.

Same methodology as tests/tune_drift_threshold_real.py: sweep threshold
candidates, evaluate precision/recall/F1 against the labeled scenarios, report
the best. This tests the SEVERITY GATE specifically — it assumes
calculate_severity()'s own internals (the per-detector floors/caps, the
diminishing-returns combination formula) are correct. If the results here look
wrong even at the best candidate threshold, that's a signal the problem is
upstream in the scoring formula itself, not just this final threshold.

Run with:
    python3 -m tests.tune_severity_threshold
"""

from monitor.scorer import calculate_severity
from tests.fixtures.severity_scenarios import SEVERITY_SCENARIOS

CANDIDATES = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def evaluate_threshold(pairs: list, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    wrong = []

    for severity, should_rollback, description in pairs:
        predicted_rollback = severity >= threshold

        if predicted_rollback and should_rollback:
            tp += 1
        elif predicted_rollback and not should_rollback:
            fp += 1
            wrong.append(("FP", severity, description))
        elif not predicted_rollback and not should_rollback:
            tn += 1
        else:
            fn += 1
            wrong.append(("FN", severity, description))

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
    print(f"Computing real severity scores for {len(SEVERITY_SCENARIOS)} labeled scenarios "
          f"via monitor.scorer.calculate_severity()...\n")

    pairs = []
    for scenario in SEVERITY_SCENARIOS:
        severity = calculate_severity(scenario["anomalies"])
        pairs.append((severity, scenario["should_rollback"], scenario["description"]))
        label = "should rollback" if scenario["should_rollback"] else "should NOT rollback"
        print(f"  severity={severity:<8} ({label:<19}) — {scenario['description'][:80]}")

    print()
    results = [evaluate_threshold(pairs, t) for t in CANDIDATES]
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
    print(f"Current value in agent/agent.py: ROLLBACK_SEVERITY_THRESHOLD = 0.8")

    if best["wrong"]:
        print(f"\nMisclassified scenarios at threshold={best['threshold']}:")
        for outcome, severity, description in best["wrong"]:
            print(f"  [{outcome}] severity={severity}  {description[:90]}")
