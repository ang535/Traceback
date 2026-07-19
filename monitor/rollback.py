MAX_ROLLBACKS_PER_TASK = 3

# Both conditions are required to trigger a full restart — see the "two-signal
# gate" note on find_rollback_point below for why severity alone isn't enough.
#
# Set via tests/fixtures/full_restart_scenarios.py + tests/
# tune_full_restart_threshold.py: 19 labeled scenarios (reasoned judgment
# calls, not measurements — full_restart has never fired in a real run),
# evaluated against the two-signal gate below. With
# MIN_ANOMALY_TYPES_FOR_FULL_RESTART in place, every threshold from 0.8 to
# 0.85 achieves F1=1.0. The gap in the labeled data sits between the
# highest-severity False case (0.7988) and the lowest-severity True case
# (0.8957). 0.85 sits inside that gap with margin on both sides.
SEVERITY_FULL_RESTART_THRESHOLD = 0.85  # combined severity must be at or above this...
MIN_ANOMALY_TYPES_FOR_FULL_RESTART = 2  # ...AND this many DISTINCT anomaly types must be firing together


def decide_rollback_strategy(anomaly_type: str, severity: float) -> str:
    """Decide which per-anomaly rollback strategy fits a given anomaly type and severity.

    This only ever returns a LOCAL strategy for one anomaly at a time — how
    far back to step for this specific failure. The decision to discard the
    entire trajectory (full_restart) is not a property of any single anomaly;
    it's made once, trajectory-wide, in find_rollback_point (see the note
    there for why that had to be pulled out of this function).

    Args:
        anomaly_type: The type of anomaly to decide a strategy for.
        severity: The combined severity score (0.0-1.0) for this step.

    Returns:
        One of "rollback_one_step" or "rollback_to_last_clean".
    """
    if anomaly_type == "infinite_loop":
        # loops are cascading by nature — going back one step just re-enters the loop
        return "rollback_to_last_clean"

    if anomaly_type == "token_explosion":
        # usually an isolated bad step — stepping back once is often enough
        return "rollback_one_step"

    if anomaly_type == "goal_drift":
        # moderate drift: one step back; severe drift: last clean step
        return "rollback_to_last_clean" if severity > 0.6 else "rollback_one_step"

    return "rollback_one_step"


def find_rollback_point(anomalies: list, trajectory: list, severity: float) -> int:
    """Find the step number to roll back to, given one or more simultaneous anomalies.

    When multiple anomalies fired at once, this finds the EARLIEST "last clean
    step" implied across all of them — i.e. the most conservative rollback
    point, since rolling back too little risks leaving a problem uncorrected.

    severity is the real combined severity computed by
    monitor.scorer.calculate_severity() (not a per-anomaly value — anomaly
    dicts from detector.py don't carry their own severity).

    Two-signal gate for full_restart: full_restart discards the ENTIRE
    trajectory, a much bigger cost than rollback_to_last_clean, which already
    reaches back exactly as far as each anomaly's own type-specific logic
    says is necessary. monitor.scorer.calculate_severity() can independently
    saturate a SINGLE anomaly (one maxed-out loop, one runaway token ratio,
    one total goal-drift departure) to severity=1.0 — identical to what
    several anomalies compounding together also produce. A severity
    threshold alone can't distinguish "one isolated failure, however
    extreme" from "multiple corroborating failures, jointly severe." Since
    rollback_to_last_clean already handles a single severe anomaly
    correctly, full_restart is gated on BOTH severity AND the number of
    distinct anomaly types firing together (MIN_ANOMALY_TYPES_FOR_FULL_RESTART),
    so an isolated single-anomaly spike doesn't trigger it regardless of
    severity.

    Args:
        anomalies: The list of anomaly dicts that fired on the current step.
        trajectory: The active trajectory so far, including the current step.
        severity: The combined severity score (0.0-1.0) for this step, as
                  computed by monitor.scorer.calculate_severity().

    Returns:
        The step_number to roll back to, or 0 for a full restart.
    """
    distinct_anomaly_types = {a["type"] for a in anomalies}
    if (severity >= SEVERITY_FULL_RESTART_THRESHOLD
            and len(distinct_anomaly_types) >= MIN_ANOMALY_TYPES_FOR_FULL_RESTART):
        return 0  # 0 means restart from nothing

    current_step_number = trajectory[-1]["step_number"]
    candidate_points = []

    for anomaly in anomalies:
        strategy = decide_rollback_strategy(anomaly["type"], severity)

        if strategy == "rollback_to_last_clean":
            # For a loop, "last clean" is the step before the repeating
            # pattern began — that's repetition_count steps back only when
            # cycle_length is 1 (a single step repeating). For a longer
            # cycle (e.g. cycle_length=2, an alternating read/run pattern),
            # the looped region spans cycle_length * repetition_count steps;
            # using repetition_count alone under-corrects by a factor of
            # cycle_length, landing the rollback inside the loop instead of
            # before it (see tests/measure_rollback_behavior.py,
            # docs/rollback_point_cycle_length_fix.md).
            repetition_count = anomaly.get("repetition_count", 1)
            cycle_length = anomaly.get("cycle_length", 1)
            loop_span = repetition_count * cycle_length
            candidate_points.append(max(0, current_step_number - loop_span))
        else:  # rollback_one_step
            candidate_points.append(max(0, current_step_number - 1))

    return min(candidate_points) if candidate_points else max(0, current_step_number - 1)


CORRECTION_TEMPLATES = {
    "goal_drift": (
        "Your previous attempt drifted away from the original task. "
        "Refocus specifically on: {original_goal}"
    ),
    "infinite_loop": (
        "Your previous attempt repeated the same action (or the same short cycle of actions, "
        "like reading then re-running the same file) multiple times without making progress. "
        "If your code already ran successfully, STOP — do not re-read or re-run it again to "
        "double-check. Report that the task is complete instead. If it has not succeeded yet, "
        "try a genuinely different approach instead of repeating the same steps."
    ),
    "token_explosion": (
        "Your previous attempt produced an unusually large response. "
        "Be more concise and focus only on the necessary change."
    ),
}


def build_combined_correction(anomalies: list, original_goal: str) -> str:
    """Build a single corrective instruction merging guidance for all triggered anomalies.

    Args:
        anomalies: The list of anomaly dicts that triggered this rollback.
        original_goal: The original task description, used to re-anchor goal drift corrections.

    Returns:
        A single string to inject as a corrective instruction when the agent resumes.
    """
    seen_types = []
    messages = []

    for anomaly in anomalies:
        anomaly_type = anomaly["type"]
        if anomaly_type in seen_types:
            continue
        seen_types.append(anomaly_type)

        template = CORRECTION_TEMPLATES.get(anomaly_type)
        if template:
            messages.append(template.format(original_goal=original_goal))

    if not messages:
        return f"Please try again, focusing on: {original_goal}"

    return " ".join(messages)


class RollbackManager:
    """Manages rollback attempts for a single task run, enforcing a retry limit."""

    def __init__(self, max_rollbacks: int = MAX_ROLLBACKS_PER_TASK):
        self.max_rollbacks = max_rollbacks
        self.rollback_count = 0
        self.rollback_history = []

    def attempt_rollback(self, anomalies: list, severity: float, trajectory: list,
                          original_goal: str, logger) -> dict:
        """Attempt a rollback, or escalate if the retry limit has been exceeded.

        Args:
            anomalies: The list of anomaly dicts that triggered this rollback.
            severity: The combined severity score for the current step.
            trajectory: The active trajectory so far, including the current step.
            original_goal: The original task description.
            logger: The TrajectoryLogger instance managing this run's steps.

        Returns:
            A dict describing the outcome: status "retrying" or "escalated",
            plus the rollback point, new branch id, and correction message
            when retrying.
        """
        if self.rollback_count >= self.max_rollbacks:
            return {
                "status": "escalated",
                "rollback_count": self.rollback_count,
                "message": (
                    f"Task has failed {self.rollback_count} times and exceeded "
                    f"the retry limit of {self.max_rollbacks}. Halting and "
                    f"escalating to user for manual intervention."
                ),
            }

        rollback_point = find_rollback_point(anomalies, trajectory, severity)
        correction = build_combined_correction(anomalies, original_goal)
        new_branch_id = logger.start_new_branch(rollback_point)

        self.rollback_count += 1
        self.rollback_history.append({
            "attempt": self.rollback_count,
            "rollback_point": rollback_point,
            "new_branch_id": new_branch_id,
            "anomaly_types": [a["type"] for a in anomalies],
            "severity": severity,
        })

        return {
            "status": "retrying",
            "attempt": self.rollback_count,
            "rollback_point": rollback_point,
            "new_branch_id": new_branch_id,
            "correction_message": correction,
        }