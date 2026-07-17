class CostTracker:
    """Tracks cumulative token spend across a task, including rollback overhead.

    total_tokens_spent counts every token ever used, including steps that were
    later discarded by a rollback. useful_tokens counts only tokens spent on
    steps that are part of the final, active trajectory. The ratio between
    them is the cost_multiplier — how much more this task cost compared to a
    single clean attempt.
    """

    def __init__(self):
        self.total_tokens_spent = 0
        self.useful_tokens = 0
        self.rollback_events = []
        self.trim_events = []

    def log_step(self, token_count: int):
        """Record tokens spent on a new step, before knowing if it will survive a rollback.

        Args:
            token_count: Tokens used by this step.
        """
        self.total_tokens_spent += token_count
        self.useful_tokens += token_count

    def log_rollback(self, discarded_steps: list):
        """Record that a GENUINE rollback (a real retry after a real problem)
        has discarded a set of previously-logged steps.

        The tokens spent on these steps remain part of total_tokens_spent
        (they were genuinely spent) but are subtracted from useful_tokens,
        since they no longer contribute to the final accepted trajectory.

        Args:
            discarded_steps: The list of step dicts being marked inactive by this rollback.
        """
        discarded_tokens = sum(step["token_count"] for step in discarded_steps)
        self.useful_tokens -= discarded_tokens
        self.rollback_events.append({
            "discarded_tokens": discarded_tokens,
            "step_count": len(discarded_steps),
        })

    def log_trim(self, discarded_steps: list):
        """Record that a success-loop early stop (agent._is_success_loop)
        has discarded redundant re-verification steps.

        Deliberately separate from log_rollback: this isn't a retry after a
        real problem, it's the system recognizing the task already succeeded
        and trimming away only the REDUNDANT extra confirmations. Before this
        distinction existed, both cases fed the same rollback_events list, so
        cost_summary()'s "rollback_count" would report 1 for a run that never
        actually rolled back or retried anything — misleading for anything
        reading the dashboard/cost summary trying to understand whether a
        real problem occurred.

        Args:
            discarded_steps: The list of step dicts being marked inactive by the trim.
        """
        discarded_tokens = sum(step["token_count"] for step in discarded_steps)
        self.useful_tokens -= discarded_tokens
        self.trim_events.append({
            "discarded_tokens": discarded_tokens,
            "step_count": len(discarded_steps),
        })

    def cost_multiplier(self) -> float:
        """Return how many times more expensive this task was versus a single clean attempt.

        A multiplier of 1.0 means no tokens were wasted on rollbacks. A
        multiplier of 2.5 means the task has cost 2.5x what the final,
        useful trajectory alone would have cost.

        Returns:
            A float >= 1.0, or 1.0 if no useful tokens have been logged yet.
        """
        if self.useful_tokens <= 0:
            return 1.0
        return self.total_tokens_spent / self.useful_tokens

    def summary(self) -> dict:
        """Return a summary of cumulative cost for display on the dashboard.

        Returns:
            A dict with total tokens spent, rollback count, tokens wasted on
            rollbacks, and the current cost multiplier.
        """
        return {
            "total_tokens": self.total_tokens_spent,
            "useful_tokens": self.useful_tokens,
            "rollback_count": len(self.rollback_events),
            "tokens_wasted_on_rollbacks": sum(e["discarded_tokens"] for e in self.rollback_events),
            "trim_count": len(self.trim_events),
            "tokens_wasted_on_trims": sum(e["discarded_tokens"] for e in self.trim_events),
            "cost_multiplier": round(self.cost_multiplier(), 2),
        }