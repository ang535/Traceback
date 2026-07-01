"""
Finds the optimal DRIFT_THRESHOLD using real computed similarity scores,
computed the SAME WAY as generate_drift_labels.py — embedding the raw
action text directly against the task, using the same cosine similarity
formula as monitor/detector.py's compute_goal_similarity.

Run with:
    python3 -m tests.tune_drift_threshold_real
"""

import numpy as np
from monitor.embeddings import get_embedding_model
from tests.fixtures.drift_scenarios_realistic import DRIFT_SCENARIOS


def compute_similarity_raw(task: str, action: str) -> float:
    """Embed task and raw action text directly — same as generate_drift_labels.py."""
    model = get_embedding_model()
    embeddings = model.encode([task, action])
    return float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )


def evaluate_threshold(pairs: list, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    wrong = []

    for score, actual_drift, action in pairs:
        predicted_drift = score < threshold

        if predicted_drift and actual_drift:
            tp += 1
        elif predicted_drift and not actual_drift:
            fp += 1
            wrong.append(("FP", score, action))
        elif not predicted_drift and not actual_drift:
            tn += 1
        else:
            fn += 1
            wrong.append(("FN", score, action))

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
    print("Computing real similarity scores (same method as generate_drift_labels.py)...")
    pairs = []
    for scenario in DRIFT_SCENARIOS:
        score = compute_similarity_raw(scenario["task"], scenario["action"])
        pairs.append((score, scenario["is_drift"], scenario["action"][:50]))
    print(f"Done. Computed {len(pairs)} scores.\n")

    candidates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = [evaluate_threshold(pairs, t) for t in candidates]
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
    print(f"Current value in detector.py: 0.4")

    if best["wrong"]:
        print(f"\nMisclassified scenarios at threshold={best['threshold']}:")
        for outcome, score, action in best["wrong"]:
            print(f"  [{outcome}] sim={score:.3f}  {action}")