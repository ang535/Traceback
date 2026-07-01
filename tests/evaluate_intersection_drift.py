"""
Evaluates the INTERSECTION variant of combined drift detection — flags drift
ONLY when BOTH check_goal_drift AND check_wrong_target_file agree — against
the same 50 realistic scenarios used by evaluate_combined_drift.py (which
uses the UNION variant: either check firing is enough).

This is a direct comparison point: same scenarios, same underlying checks,
only the combination RULE differs (AND vs OR). Comparing this script's
output against evaluate_combined_drift.py's output reveals the real
precision/recall tradeoff between the two combination strategies.

Run with:
    python3 -m tests.evaluate_intersection_drift
"""

from monitor.detector import check_goal_drift, check_wrong_target_file, compute_goal_similarity
from tests.fixtures.drift_scenarios_realistic import DRIFT_SCENARIOS
from tests.evaluate_combined_drift import build_fake_step


def evaluate():
    tp = fp = tn = fn = 0
    detail = []

    for scenario in DRIFT_SCENARIOS:
        task = scenario["task"]
        action = scenario["action"]
        actual_drift = scenario["is_drift"]

        fake_step = build_fake_step(action)
        fake_trajectory = [fake_step]

        similarity_score = compute_goal_similarity(task, fake_step)

        drift_result = check_goal_drift(task, fake_step, fake_trajectory, similarity_score=similarity_score)
        wrong_file_result = check_wrong_target_file(task, fake_step, similarity_score=similarity_score)

        # INTERSECTION: both checks must agree for this to count as drift
        predicted_drift = (drift_result is not None) and (wrong_file_result is not None)

        if predicted_drift and actual_drift:
            outcome = "TP"
            tp += 1
        elif predicted_drift and not actual_drift:
            outcome = "FP"
            fp += 1
        elif not predicted_drift and not actual_drift:
            outcome = "TN"
            tn += 1
        else:
            outcome = "FN"
            fn += 1

        detail.append({
            "action": action[:55],
            "outcome": outcome,
            "similarity": round(similarity_score, 3),
            "drift_flag": drift_result is not None,
            "wrong_file_flag": wrong_file_result is not None,
            "actual_drift": actual_drift,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n{'Outcome':<10}{'Sim':<8}{'Drift':<8}{'WrongFile':<11}{'Actual':<8}Action")
    print("-" * 85)
    for d in detail:
        flag = " <-- WRONG" if d["outcome"] in ("FP", "FN") else ""
        print(
            f"{d['outcome']:<10}{d['similarity']:<8}{str(d['drift_flag']):<8}"
            f"{str(d['wrong_file_flag']):<11}{str(d['actual_drift']):<8}{d['action']}{flag}"
        )

    print(f"\n{'='*60}")
    print(f"INTERSECTION RESULT (check_goal_drift AND check_wrong_target_file)")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision: {round(precision, 3)}   Recall: {round(recall, 3)}   F1: {round(f1, 3)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate()