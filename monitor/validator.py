import re


def validate_task(task: str) -> dict:
    """Check a task description for obvious problems before running the agent.

    This is a cheap, rule-based check — no LLM call involved. It catches
    empty input, suspiciously short input, and input with no file reference
    at all, before any tokens are spent.

    Args:
        task: The user's task description string.

    Returns:
        A dict with "valid" (bool) and "reason" (str or None).
    """
    task = task.strip()

    if not task:
        return {"valid": False, "reason": "Task description is empty."}

    if len(task) < 10:
        return {"valid": False, "reason": "Task description is too short to be actionable."}

    # look for something that resembles a filename: word characters, optional
    # path separators, a dot, and an extension (e.g. tasks/buggy_add.py)
    file_pattern = r"[\w/\-]+\.\w+"
    has_file_reference = re.search(file_pattern, task) is not None

    if not has_file_reference:
        return {
            "valid": False,
            "reason": (
                "No specific file was mentioned in the task. "
                "Please name the file to work on, e.g. 'Fix the bug in buggy_add.py'."
            ),
        }

    return {"valid": True, "reason": None}


def check_first_step_validity(first_step: dict) -> dict:
    """Halt immediately if the agent's first action was reading a file that doesn't exist.

    This is a second, cheap layer of defense beyond validate_task — it catches
    the case where the task text looked fine (it named a file) but that file
    doesn't actually exist on disk.

    Args:
        first_step: A dict describing the first logged step, expected to have
                    "tool_used" and "output_summary" keys.

    Returns:
        A dict with "status" ("ok" or "halted") and "reason" (str or None).
    """
    tool_used = first_step.get("tool_used", "")
    output = first_step.get("output_summary", "")

    if tool_used == "read_file" and "not found" in output.lower():
        return {
            "status": "halted",
            "reason": f"The file referenced in the task does not exist: {output}",
        }

    return {"status": "ok", "reason": None}