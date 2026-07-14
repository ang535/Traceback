"""Labeled scenarios for tuning ROLLBACK_SEVERITY_THRESHOLD.

Unlike drift/loop/token tuning, this doesn't need any live agent runs or LLM
calls — monitor/scorer.py's calculate_severity() is pure Python, so we can
feed it realistic anomaly-dict combinations directly and get back the exact
severity score the real system would compute.

Each scenario is a list of anomaly dicts (the same shape run_all_detectors()
produces) plus a human judgment: should this combination have triggered a
rollback? Labeling philosophy, decided explicitly rather than assumed:

- A completed-but-redundant loop (agent already succeeded, then keeps
  re-verifying instead of stopping) DOES count as should_rollback=True.
  Wasted tokens/steps are a real cost even when the final output is
  correct — catching wasteful trajectories, not just wrong ones, is the
  whole premise of this project.
- A single anomaly at exactly its detector's minimum trigger point is
  still labeled should_rollback=True if intervening on it matches that
  philosophy (e.g. a bare-minimum infinite_loop) — the detector's own
  threshold was already empirically validated to only fire on genuine
  problems, so passing that bar at all is meaningful.
- A single low-confidence or borderline signal alone is generally labeled
  should_rollback=False — the system is deliberately built to not overreact
  to one noisy data point (see confidence="low" and LOW_CONFIDENCE_PENALTY).
- Multiple anomalies firing simultaneously, even individually moderate
  ones, generally push toward should_rollback=True — corroborating signals
  are stronger evidence than any one of them alone.

Where real numbers were available (scenario "real_success_loop"), they're
taken directly from an actual live dashboard run rather than invented.
"""

SEVERITY_SCENARIOS = [
    # ---- single goal_drift ----
    {
        "description": "goal_drift, similarity=0.55 — just below the 0.6 cutoff, barely a drift",
        "anomalies": [
            {"type": "goal_drift", "step": 4, "similarity_score": 0.55, "confidence": "normal"},
        ],
        "should_rollback": False,
    },
    {
        "description": "goal_drift, similarity=0.3 — moderate drift, single signal only",
        "anomalies": [
            {"type": "goal_drift", "step": 5, "similarity_score": 0.3, "confidence": "normal"},
        ],
        "should_rollback": False,
    },
    {
        "description": "goal_drift, similarity=0.05 — near-total departure from the task",
        "anomalies": [
            {"type": "goal_drift", "step": 6, "similarity_score": 0.05, "confidence": "normal"},
        ],
        "should_rollback": True,
    },
    {
        "description": "goal_drift, similarity=0.4, low confidence (step 1) — moderate but noisy",
        "anomalies": [
            {"type": "goal_drift", "step": 1, "similarity_score": 0.4, "confidence": "low"},
        ],
        "should_rollback": False,
    },
    {
        "description": "goal_drift, similarity=0.0, low confidence (step 1) — total mismatch but still just one noisy data point",
        "anomalies": [
            {"type": "goal_drift", "step": 1, "similarity_score": 0.0, "confidence": "low"},
        ],
        "should_rollback": False,
    },

    # ---- single infinite_loop ----
    {
        "description": "infinite_loop, exactly 3 repetitions (bare minimum trigger)",
        "anomalies": [
            {"type": "infinite_loop", "step": 8, "repetition_count": 3},
        ],
        "should_rollback": True,
    },
    {
        "description": "infinite_loop, 4 repetitions",
        "anomalies": [
            {"type": "infinite_loop", "step": 9, "repetition_count": 4},
        ],
        "should_rollback": True,
    },
    {
        "description": "infinite_loop, 6 repetitions (severity cap)",
        "anomalies": [
            {"type": "infinite_loop", "step": 12, "repetition_count": 6},
        ],
        "should_rollback": True,
    },

    # ---- single token_explosion ----
    {
        "description": "token_explosion, ratio=2.2 (bare minimum trigger) — could just be a longer legitimate response",
        "anomalies": [
            {"type": "token_explosion", "step": 5, "token_count": 900, "rolling_average": 409,
             "ratio": 2.2, "detection_method": "rolling_average"},
        ],
        "should_rollback": False,
    },
    {
        "description": "token_explosion, ratio=6.0 — clearly excessive relative to the run's own history",
        "anomalies": [
            {"type": "token_explosion", "step": 7, "token_count": 2400, "rolling_average": 400,
             "ratio": 6.0, "detection_method": "rolling_average"},
        ],
        "should_rollback": True,
    },
    {
        "description": "token_explosion, ratio=10.0 (severity cap) — runaway generation",
        "anomalies": [
            {"type": "token_explosion", "step": 3, "token_count": 4000, "rolling_average": 400,
             "ratio": 10.0, "detection_method": "rolling_average"},
        ],
        "should_rollback": True,
    },
    {
        "description": "token_explosion via absolute_ceiling (single step over 4000 tokens, no rolling average yet)",
        "anomalies": [
            {"type": "token_explosion", "step": 1, "token_count": 4500, "ceiling": 4000,
             "detection_method": "absolute_ceiling"},
        ],
        "should_rollback": True,
    },

    # ---- single wrong_target_file ----
    {
        "description": "wrong_target_file alone — agent edited a file never named in the task",
        "anomalies": [
            {"type": "wrong_target_file", "step": 2, "expected_files": ["buggy_add.py"],
             "actual_file": "buggy_multiply.py", "similarity_score": 0.5},
        ],
        "should_rollback": True,
    },

    # ---- real scenario, taken directly from an actual dashboard run ----
    {
        "description": (
            "REAL scenario from a live dashboard run: agent already fixed "
            "tasks/buggy_multiply.py, then looped read_file -> run_code "
            "re-verifying the already-correct file. infinite_loop fired at "
            "exactly 3 repetitions alongside a token_explosion at ratio=2.83 "
            "on the same step (step 11 of that run)."
        ),
        "anomalies": [
            {"type": "infinite_loop", "step": 11, "repetition_count": 3},
            {"type": "token_explosion", "step": 11, "token_count": 1461, "rolling_average": 516.9,
             "ratio": 2.8264654672083576, "detection_method": "rolling_average"},
        ],
        "should_rollback": True,
    },

    # ---- multi-anomaly combinations ----
    {
        "description": "goal_drift(0.5, normal) + token_explosion(ratio=3.0) — two moderate, corroborating signals",
        "anomalies": [
            {"type": "goal_drift", "step": 6, "similarity_score": 0.5, "confidence": "normal"},
            {"type": "token_explosion", "step": 6, "token_count": 1200, "rolling_average": 400,
             "ratio": 3.0, "detection_method": "rolling_average"},
        ],
        "should_rollback": True,
    },
    {
        "description": "wrong_target_file + goal_drift(0.55, low confidence) — wrong file alone already qualifies, drift adds weak corroboration",
        "anomalies": [
            {"type": "wrong_target_file", "step": 1, "expected_files": ["buggy_add.py"],
             "actual_file": "unrelated_module.py", "similarity_score": 0.3},
            {"type": "goal_drift", "step": 1, "similarity_score": 0.55, "confidence": "low"},
        ],
        "should_rollback": True,
    },
    {
        "description": "infinite_loop(3) + token_explosion(2.2) + goal_drift(0.5, normal) — three independent signals at once",
        "anomalies": [
            {"type": "infinite_loop", "step": 10, "repetition_count": 3},
            {"type": "token_explosion", "step": 10, "token_count": 900, "rolling_average": 409,
             "ratio": 2.2, "detection_method": "rolling_average"},
            {"type": "goal_drift", "step": 10, "similarity_score": 0.5, "confidence": "normal"},
        ],
        "should_rollback": True,
    },
    {
        "description": "goal_drift(0.58, normal, barely below cutoff) + token_explosion(2.25, barely above minimum) — two very marginal signals",
        "anomalies": [
            {"type": "goal_drift", "step": 4, "similarity_score": 0.58, "confidence": "normal"},
            {"type": "token_explosion", "step": 4, "token_count": 920, "rolling_average": 409,
             "ratio": 2.25, "detection_method": "rolling_average"},
        ],
        "should_rollback": False,
    },
    {
        "description": "infinite_loop(6, cap) + goal_drift(0.1, normal) — both individually severe",
        "anomalies": [
            {"type": "infinite_loop", "step": 15, "repetition_count": 6},
            {"type": "goal_drift", "step": 15, "similarity_score": 0.1, "confidence": "normal"},
        ],
        "should_rollback": True,
    },
    {
        "description": "two low-confidence, moderate goal_drift readings on early steps (steps 1 and 2), nothing else",
        "anomalies": [
            {"type": "goal_drift", "step": 1, "similarity_score": 0.45, "confidence": "low"},
        ],
        "should_rollback": False,
    },
]
