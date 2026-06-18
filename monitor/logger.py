import time


def create_step_entry(step_number, branch_id, tool_used, input_summary,
                       output_summary, token_count):
    """Create a single trajectory step entry.

    Every step the agent takes gets recorded in this exact shape. The
    branch_id and is_active fields exist to support rollback later: when a
    rollback happens, old steps are marked is_active=False rather than
    deleted, so the full history is preserved for debugging while only the
    "active" branch is treated as the current trajectory.

    Args:
        step_number: The position of this step in its branch (1, 2, 3, ...).
        branch_id: Which branch this step belongs to. Starts at 0, increments
                   by 1 every time a rollback creates a new branch.
        tool_used: The name of the tool called at this step (e.g. "read_file"),
                   or "reasoning" if the agent produced text with no tool call.
        input_summary: A short description of what was passed into the tool.
        output_summary: A short description of what the tool returned.
        token_count: How many tokens this step consumed.

    Returns:
        A dict representing this step.
    """
    return {
        "step_number": step_number,
        "branch_id": branch_id,
        "is_active": True,
        "tool_used": tool_used,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "token_count": token_count,
        "timestamp": time.time(),
    }


class TrajectoryLogger:
    """Records every step of an agent run as a flat list with branch tracking."""

    def __init__(self):
        self.log = []
        self.current_branch_id = 0
        self._next_step_number = 1

    def log_step(self, tool_used, input_summary, output_summary, token_count):
        """Record one new step in the currently active branch."""
        entry = create_step_entry(
            step_number=self._next_step_number,
            branch_id=self.current_branch_id,
            tool_used=tool_used,
            input_summary=input_summary,
            output_summary=output_summary,
            token_count=token_count,
        )
        self.log.append(entry)
        self._next_step_number += 1
        return entry

    def get_active_trajectory(self):
        """Return only the steps that are part of the current, live branch."""
        return [entry for entry in self.log if entry["is_active"]]

    def get_full_log(self):
        """Return every step ever recorded, including superseded branches."""
        return self.log

    def start_new_branch(self, rollback_to_step):
        """Mark steps after rollback_to_step as inactive and begin a new branch.

        Called by the rollback manager. Steps with step_number greater than
        rollback_to_step in the current branch get is_active=False — they are
        not deleted, just excluded from the active trajectory going forward.

        Args:
            rollback_to_step: The step_number to roll back to. Steps after
                              this one in the current branch are superseded.

        Returns:
            The new branch_id.
        """
        for entry in self.log:
            if entry["branch_id"] == self.current_branch_id and entry["step_number"] > rollback_to_step:
                entry["is_active"] = False

        self.current_branch_id += 1
        self._next_step_number = rollback_to_step + 1
        return self.current_branch_id