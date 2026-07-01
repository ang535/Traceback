"""
STEP 2: Run the real embedding model against the plain-English scenarios in
tests/fixtures/drift_scenarios_realistic.py, computing the REAL similarity
score for each one using the exact same model detector.py uses.

This produces real (similarity_score, is_drift) pairs — actual measurements
paired with your real judgment — which is the only honest input for tuning
DRIFT_THRESHOLD.

Run with:
    python3 -m tests.generate_drift_labels
"""

import numpy as np
from monitor.embeddings import get_embedding_model
from tests.fixtures.drift_scenarios_realistic import DRIFT_SCENARIOS


def compute_similarity(task: str, action: str) -> float:
    model = get_embedding_model()
    embeddings = model.encode([task, action])
    similarity = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
    return similarity


if __name__ == "__main__":
    print(f"\n{'Similarity':<12}{'Your Label':<14}Action")
    print("-" * 70)

    computed_pairs = []
    for scenario in DRIFT_SCENARIOS:
        score = compute_similarity(scenario["task"], scenario["action"])
        label = "DRIFT" if scenario["is_drift"] else "fine"
        computed_pairs.append({"similarity_score": round(score, 4), "is_drift": scenario["is_drift"]})
        print(f"{score:<12.4f}{label:<14}{scenario['action'][:55]}")

    print(f"\n{'='*70}")
    print("Copy the block below into tests/fixtures/drift_threshold_scenarios.py")
    print("This replaces the placeholder values with REAL computed scores.")
    print(f"{'='*70}\n")

    print("DRIFT_THRESHOLD_SCENARIOS = [")
    for pair in computed_pairs:
        print(f'    {{"similarity_score": {pair["similarity_score"]}, "is_drift": {pair["is_drift"]}}},')
    print("]")