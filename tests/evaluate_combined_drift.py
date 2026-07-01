"""
Evaluates the COMBINED drift detection — embedding similarity check
(check_goal_drift) PLUS the softened deterministic check
(check_wrong_target_file) — against the 50 realistic scenarios, to see
whether adding the new check actually resolves the overlap problem found
when using embedding similarity alone.

check_wrong_target_file now only fires when BOTH signals agree: the file is
not named in the task AND similarity is below RELATED_FILE_LENIENCY_THRESHOLD.
This mirrors exactly how run_all_detectors uses them together in production —
computing similarity once and sharing it between both checks.

For each scenario:
  - Computes the real similarity score ONCE
  - Runs check_goal_drift using that score
  - Runs check_wrong_target_file using that SAME score
  - "predicted_drift" is True if EITHER detector fires
  - Compares against your is_drift label
  - Reports precision/recall/F1 for the combined approach

Run with:
    python3 -m tests.evaluate_combined_drift
"""

from monitor.detector import check_goal_drift, check_wrong_target_file, compute_goal_similarity
from tests.fixtures.drift_scenarios_realistic import DRIFT_SCENARIOS


def build_fake_step(action_text: str, step_number: int = 2) -> dict:
    """Reconstruct a minimal step dict from the plain-English action string,
    extracting a filepath if the action mentions one with a recognizable verb."""
    if "Reading the file " in action_text:
        tool = "read_file"
        filepath = action_text.split("Reading the file ")[1].split(" to")[0]
    elif "Writing changes to the file " in action_text:
        tool = "write_file"
        filepath = action_text.split("Writing changes to the file ")[1].split(" which")[0].split(",")[0]
    elif "Writing a new test file " in action_text:
        tool = "write_file"
        filepath = action_text.split("Writing a new test file ")[1].split(" to")[0]
    elif "Running" in action_text and "the file " in action_text:
        tool = "run_code"
        filepath = action_text.split("the file ")[1].split(" to")[0]
    else:
        tool = "other"
        filepath = None

    return {
        "step_number": step_number,
        "tool_used": tool,
        "input_summary": {"filepath": filepath} if filepath else action_text,
        "output_summary": "",
    }


def evaluate():
    tp = fp = tn = fn = 0
    detail = []

    for scenario in DRIFT_SCENARIOS:
        task = scenario["task"]
        action = scenario["action"]
        actual_drift = scenario["is_drift"]

        fake_step = build_fake_step(action)
        fake_trajectory = [fake_step]

        # compute similarity ONCE, shared between both checks — matches how
        # run_all_detectors actually does it in production
        similarity_score = compute_goal_similarity(task, fake_step)

        drift_result = check_goal_drift(task, fake_step, fake_trajectory, similarity_score=similarity_score)
        wrong_file_result = check_wrong_target_file(task, fake_step, similarity_score=similarity_score)

        predicted_drift = (drift_result is not None) or (wrong_file_result is not None)

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
    print(f"COMBINED RESULT (check_goal_drift OR check_wrong_target_file)")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision: {round(precision, 3)}   Recall: {round(recall, 3)}   F1: {round(f1, 3)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate()