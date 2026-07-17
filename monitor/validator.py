import re


# Reasoned, not swept — this isn't a detection threshold with a signal to
# sweep against, it's an input-validation floor with no live-trial data to
# check it against. Checked by hand instead: the old value (10) rejects
# real, clearly-actionable short task phrasings like "fix x.py" or
# "run x.py" (8 characters each) with the misleading message "too short to
# be actionable" — they're not; they're just terse and already name a real
# file. Lowered to 6, chosen as the smallest floor that still blocks
# degenerate junk that would otherwise slip past the file-reference regex
# below purely by accident (e.g. "a.b" or "ab.c", 3-4 characters, which
# technically match `[\w/\-]+\.\w+` despite not being a real task). 6 keeps
# both properties: "fix x.py"/"run x.py" (8 chars) now correctly pass,
# while "a.b"/"ab.c" (3-4 chars) still correctly get rejected here, before
# ever reaching the file-reference check where they'd otherwise wrongly
# pass as "valid" due to the accidental regex match.
MIN_TASK_LENGTH = 6


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

    if len(task) < MIN_TASK_LENGTH:
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