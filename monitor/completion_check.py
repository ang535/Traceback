def verify_task_completion(trajectory: list) -> dict:
    """Verify task completion using actual logged data, not the agent's self-report.

    "Done" is defined as: the most recent run_code call in the active
    trajectory exited without an error. If the agent never called run_code
    at all, or the last call still failed, completion is not verified.

    Args:
        trajectory: The active trajectory (already filtered to is_active=True).

    Returns:
        A dict with "verified" (bool) and "reason" (str or None).
    """
    if not trajectory:
        return {"verified": False, "reason": "No steps were taken."}

    last_run_step = None
    for entry in reversed(trajectory):
        if entry["tool_used"] == "run_code":
            last_run_step = entry
            break

    if last_run_step is None:
        return {
            "verified": False,
            "reason": "The agent never ran the code to verify its fix.",
        }

    output = str(last_run_step.get("output_summary", ""))
    if output.lower().startswith("error"):
        return {
            "verified": False,
            "reason": f"The last code execution still failed: {output[:200]}",
        }

    return {"verified": True, "reason": None}


def finalize_run(trajectory: list, agent_final_message: str, rollback_status: str = None,
                  escalation_reason: str = None) -> dict:
    """Determine the final status of a task run.

    Three possible outcomes, each meaningfully different and shown distinctly
    on the dashboard:
      - "escalated": the rollback manager hit its retry limit, or the run was
                      halted directly for a reason retrying can't fix. Needs
                      manual intervention.
      - "unverified": the agent believes it finished, but verify_task_completion disagrees.
      - "success": the agent finished AND the last run_code call exited clean.

    Args:
        trajectory: The active trajectory at the end of the run.
        agent_final_message: The agent's own final natural-language response.
        rollback_status: "escalated" if the rollback manager exceeded its
                          retry limit during this run, otherwise None.
        escalation_reason: A specific explanation for why the run escalated,
                            when the cause is more specific than exhausting
                            the retry budget (e.g. an unproductive read-only
                            loop — see agent.agent._is_unproductive_read_loop).
                            Falls back to the generic retry-limit message if
                            not given.

    Returns:
        A dict with "status", "message", and "warning" (only present if unverified).
    """
    if rollback_status == "escalated":
        return {
            "status": "escalated",
            "message": agent_final_message,
            "warning": escalation_reason or "Task exceeded the maximum number of automatic retries.",
        }

    verification = verify_task_completion(trajectory)

    if verification["verified"]:
        return {"status": "success", "message": agent_final_message}

    return {
        "status": "unverified",
        "message": agent_final_message,
        "warning": verification["reason"],
    }