"""Labeled scenarios for tuning SEVERITY_FULL_RESTART_THRESHOLD.

Same offline methodology as severity_scenarios.py: feed real anomaly-dict
combinations into monitor.scorer.calculate_severity() and compare the actual
severity number against a human judgment call. No live agent runs needed.

IMPORTANT CAVEAT (read before trusting these numbers the way TOKEN_MIN_RATIO's
were trusted): full_restart has never actually fired in any real run of this
system — find_rollback_point() was silently defaulting every anomaly's
severity to 0.5 until the bug fix that motivated this file, so
SEVERITY_FULL_RESTART_THRESHOLD (0.85) has been dead code since it was
written. Unlike TOKEN_MIN_RATIO (measured from real trial runs) or
ROLLBACK_SEVERITY_THRESHOLD (scenarios grounded in one real dashboard run),
there is no real full_restart incident to anchor these labels to. Every label
below is a reasoned judgment call, not a measurement. Treat the tuned value
as a documented starting point, not a validated one — revisit once real
full_restart events exist to check against.

Labeling philosophy:

- full_restart discards the ENTIRE trajectory (branch point 0) and eats every
  token spent so far as pure waste. rollback_to_last_clean already reaches
  back as far as the anomaly's own type-specific logic says is necessary
  (e.g. repetition_count steps back for a loop). Given that cost gap,
  full_restart should be reserved for cases where there's reason to distrust
  the trajectory as a whole, not just its most recent steps.
- A SINGLE anomaly, however severe in its own detector's terms (a loop
  repeated 10 times, a total goal_drift departure, a runaway token ratio of
  20x), is still just one localized failure mode. rollback_to_last_clean/
  rollback_one_step already targets it correctly. These are labeled
  should_full_restart=False even when calculate_severity() saturates them to
  1.0 — see the KNOWN LIMITATION note below.
- MULTIPLE distinct anomaly types firing together at meaningfully high
  severity are labeled True — that's actual corroborating evidence that the
  whole approach broke down, not just the last few steps.

STRUCTURAL LIMITATION FOUND, THEN FIXED: a single maxed-out anomaly and a
genuine multi-signal compound failure can both compute to severity=1.0 via
calculate_severity() — a scalar severity threshold alone cannot tell them
apart. The first pass at tuning against this scenario set surfaced that gap
directly (best achievable F1 was only 0.706, with 4 of 5 misclassifications
being exactly this ambiguous severity=1.0 band). Rather than accept that
ceiling, monitor.rollback.find_rollback_point was changed to gate full_restart
on TWO signals: severity >= SEVERITY_FULL_RESTART_THRESHOLD AND distinct
anomaly type count >= MIN_ANOMALY_TYPES_FOR_FULL_RESTART. All labels below are
written against that two-signal gate.
"""

FULL_RESTART_SCENARIOS = [
    # ---- single anomaly, low/moderate magnitude ----
    {
        "description": "single infinite_loop, bare minimum (3 reps)",
        "anomalies": [{"type": "infinite_loop", "repetition_count": 3}],
        "should_full_restart": False,
    },
    {
        "description": "single token_explosion, bare minimum ratio (2.2)",
        "anomalies": [{"type": "token_explosion", "ratio": 2.2, "detection_method": "rolling_average"}],
        "should_full_restart": False,
    },
    {
        "description": "single goal_drift, moderate (similarity=0.3)",
        "anomalies": [{"type": "goal_drift", "similarity_score": 0.3, "confidence": "normal"}],
        "should_full_restart": False,
    },
    {
        "description": "single wrong_target_file (flat 0.7)",
        "anomalies": [{"type": "wrong_target_file"}],
        "should_full_restart": False,
    },
    {
        "description": "single token_explosion via absolute_ceiling (no rolling average yet)",
        "anomalies": [{"type": "token_explosion", "detection_method": "absolute_ceiling"}],
        "should_full_restart": False,
    },

    # ---- single anomaly, maxed out (this is the known-limitation band) ----
    {
        "description": "single infinite_loop, 6 reps (severity cap) — one isolated loop, however long",
        "anomalies": [{"type": "infinite_loop", "repetition_count": 6}],
        "should_full_restart": False,
    },
    {
        "description": "single infinite_loop, 10 reps (well past cap) — still one isolated loop",
        "anomalies": [{"type": "infinite_loop", "repetition_count": 10}],
        "should_full_restart": False,
    },
    {
        "description": "single token_explosion, ratio=10 (severity cap) — one runaway step",
        "anomalies": [{"type": "token_explosion", "ratio": 10.0, "detection_method": "rolling_average"}],
        "should_full_restart": False,
    },
    {
        "description": "single goal_drift, similarity=0.0 (total departure) — one severe but isolated signal",
        "anomalies": [{"type": "goal_drift", "similarity_score": 0.0, "confidence": "normal"}],
        "should_full_restart": False,
    },

    # ---- two anomalies, both moderate (not enough to condemn the whole trajectory) ----
    {
        "description": "infinite_loop(4) + token_explosion(ratio=4.0) — two moderate signals",
        "anomalies": [
            {"type": "infinite_loop", "repetition_count": 4},
            {"type": "token_explosion", "ratio": 4.0, "detection_method": "rolling_average"},
        ],
        "should_full_restart": False,
    },
    {
        "description": "wrong_target_file + token_explosion(absolute_ceiling) — two signals, still moderate combined",
        "anomalies": [
            {"type": "wrong_target_file"},
            {"type": "token_explosion", "detection_method": "absolute_ceiling"},
        ],
        "should_full_restart": False,
    },
    {
        "description": "goal_drift(0.3) + token_explosion(ratio=3.0) — two moderate, corroborating signals",
        "anomalies": [
            {"type": "goal_drift", "similarity_score": 0.3, "confidence": "normal"},
            {"type": "token_explosion", "ratio": 3.0, "detection_method": "rolling_average"},
        ],
        "should_full_restart": False,
    },

    # ---- multiple anomalies compounding to high severity (genuine corroboration) ----
    {
        "description": "infinite_loop(5) + token_explosion(ratio=7.0) + goal_drift(0.2) — three distinct failure modes at once",
        "anomalies": [
            {"type": "infinite_loop", "repetition_count": 5},
            {"type": "token_explosion", "ratio": 7.0, "detection_method": "rolling_average"},
            {"type": "goal_drift", "similarity_score": 0.2, "confidence": "normal"},
        ],
        "should_full_restart": True,
    },
    {
        "description": "infinite_loop(6, cap) + token_explosion(10, cap) — two independently maxed-out failure modes",
        "anomalies": [
            {"type": "infinite_loop", "repetition_count": 6},
            {"type": "token_explosion", "ratio": 10.0, "detection_method": "rolling_average"},
        ],
        "should_full_restart": True,
    },
    {
        "description": "infinite_loop(6, cap) + token_explosion(10, cap) + goal_drift(0.0) — three independently maxed-out failure modes",
        "anomalies": [
            {"type": "infinite_loop", "repetition_count": 6},
            {"type": "token_explosion", "ratio": 10.0, "detection_method": "rolling_average"},
            {"type": "goal_drift", "similarity_score": 0.0, "confidence": "normal"},
        ],
        "should_full_restart": True,
    },
    {
        "description": "wrong_target_file + infinite_loop(6, cap) — wrong file AND a maxed-out loop",
        "anomalies": [
            {"type": "wrong_target_file"},
            {"type": "infinite_loop", "repetition_count": 6},
        ],
        "should_full_restart": True,
    },
    {
        "description": "wrong_target_file + goal_drift(0.0, total departure) — wrong file AND total drift",
        "anomalies": [
            {"type": "wrong_target_file"},
            {"type": "goal_drift", "similarity_score": 0.0, "confidence": "normal"},
        ],
        "should_full_restart": True,
    },
    {
        "description": "wrong_target_file + infinite_loop(4) + token_explosion(ratio=4.0) — three distinct types, but combined severity stays moderate (0.80) — not enough on its own",
        "anomalies": [
            {"type": "wrong_target_file"},
            {"type": "infinite_loop", "repetition_count": 4},
            {"type": "token_explosion", "ratio": 4.0, "detection_method": "rolling_average"},
        ],
        "should_full_restart": False,
    },
    {
        "description": "wrong_target_file + infinite_loop(5) + token_explosion(ratio=6.0) — three distinct types, none individually maxed, but combined severity is genuinely high (0.90)",
        "anomalies": [
            {"type": "wrong_target_file"},
            {"type": "infinite_loop", "repetition_count": 5},
            {"type": "token_explosion", "ratio": 6.0, "detection_method": "rolling_average"},
        ],
        "should_full_restart": True,
    },
]
