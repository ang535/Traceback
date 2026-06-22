MAX_ROLLBACKS_PER_TASK = 3

SEVERITY_FULL_RESTART_THRESHOLD = 0.85  # at or above this, even one anomaly warrants a full restart


def decide_rollback_strategy(anomaly_type: str, severity: float) -> str:
    """Decide which rollback strategy fits a given anomaly type and severity.

    Args:
        anomaly_type: The type of the (most severe) anomaly that triggered rollback.
        severity: The combined severity score (0.0-1.0) for this step.

    Returns:
        One of "rollback_one_step", "rollback_to_last_clean", or "full_restart".
    """
    if severity >= SEVERITY_FULL_RESTART_THRESHOLD:
        return "full_restart"

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


def find_rollback_point(anomalies: list, trajectory: list) -> int:
    """Find the step number to roll back to, given one or more simultaneous anomalies.

    When multiple anomalies fired at once, this finds the EARLIEST "last clean
    step" implied across all of them — i.e. the most conservative rollback
    point, since rolling back too little risks leaving a problem uncorrected.

    Args:
        anomalies: The list of anomaly dicts that fired on the current step.
        trajectory: The active trajectory so far, including the current step.

    Returns:
        The step_number to roll back to.
    """
    current_step_number = trajectory[-1]["step_number"]
    candidate_points = []

    for anomaly in anomalies:
        strategy = decide_rollback_strategy(anomaly["type"], anomaly.get("severity", 0.5))

        if strategy == "full_restart":
            candidate_points.append(0)  # 0 means restart from nothing
        elif strategy == "rollback_to_last_clean":
            # for a loop, "last clean" is the step before the repetition began
            repetition_count = anomaly.get("repetition_count", 1)
            candidate_points.append(max(0, current_step_number - repetition_count))
        else:  # rollback_one_step
            candidate_points.append(max(0, current_step_number - 1))

    return min(candidate_points) if candidate_points else max(0, current_step_number - 1)


CORRECTION_TEMPLATES = {
    "goal_drift": (
        "Your previous attempt drifted away from the original task. "
        "Refocus specifically on: {original_goal}"
    ),
    "infinite_loop": (
        "Your previous attempt repeated the same action multiple times without making progress. "
        "Try a different approach instead of repeating the last action."
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

        rollback_point = find_rollback_point(anomalies, trajectory)
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