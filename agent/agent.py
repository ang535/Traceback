import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from agent.tools import TOOLS
from monitor.validator import validate_task, check_first_step_validity
from monitor.logger import TrajectoryLogger
from monitor.detector import run_all_detectors
from monitor.scorer import calculate_severity
from monitor.rollback import RollbackManager
from monitor.cost_tracker import CostTracker
from monitor.completion_check import finalize_run

load_dotenv()

# Which LLM backend build_agent() uses. "groq" is the default for testing —
# Groq's free tier (1000 req/day) supports iterating on eval/tuning scripts
# without hitting Gemini's 20 req/day free-tier ceiling. Switch to "gemini"
# for production runs.
PROVIDER = "groq"

# llama-3.3-70b-versatile repeatedly emitted malformed, legacy-style function
# call tags (<function=name{args}</function>) that Groq's API rejects with a
# 400, even at temperature=0. Groq's own docs recommend the newer GPT-OSS
# models for tool use — openai/gpt-oss-20b has full local tool-calling
# support, is faster (~1000 t/s vs ~280 t/s), and is cheaper per token.
GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-2.5-flash"

# Set at 0.45 via tests/fixtures/severity_scenarios.py +
# tests/tune_severity_threshold.py: 20 labeled anomaly-combination scenarios
# run through the actual, unmodified monitor.scorer.calculate_severity().
# 0.45 gives the best F1 with perfect recall (1.0) — it catches every
# scenario labeled should_rollback, including an isolated bare-minimum
# infinite_loop (severity 0.5). F1 at 0.45 is 0.963 (after the
# LOW_CONFIDENCE_PENALTY fix in monitor/scorer.py; was 0.929 before). The one
# remaining false positive is a single NORMAL-confidence moderate goal_drift
# reading with no other anomaly backing it up.
ROLLBACK_SEVERITY_THRESHOLD = 0.45

# Grounded via tests/analyze_step_budget.py against every real trial's step
# count collected across this project. This is a last-resort circuit
# breaker, not the primary control — the anomaly detectors + rollback system
# are meant to catch a genuinely stuck trajectory well before a step-count
# cap matters. Widest real legitimate task difficulty tested so far: 11
# steps (the deliberately larger "write 25 test functions" structural task);
# trivial bug-fix tasks take 2-3 steps. 25 gives 2.27x headroom over the
# hardest real task seen. Caveat: no real trial has exercised a genuinely
# complex, multi-file task in the 12-24 step range, so this confirms margin
# over everything tested but isn't a swept/validated optimum the way
# TOKEN_MIN_RATIO or ROLLBACK_SEVERITY_THRESHOLD are.
MAX_TOTAL_STEPS = 25  # hard safety cap, independent of rollback retry limits


def build_agent():
    """Build and return the LangGraph coding agent, powered by PROVIDER."""
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            # llama-3.3-70b-versatile occasionally emits a malformed,
            # legacy-style function-call tag (<function=name{args}</function>)
            # instead of a proper structured tool call, which Groq's API then
            # rejects with a 400. This happens more often at higher sampling
            # temperature; temperature=0 makes tool-call formatting far more
            # reliable at the cost of response diversity we don't need here.
            temperature=0,
        )
    elif PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}. Expected 'groq' or 'gemini'.")

    agent = create_react_agent(llm, TOOLS)
    return agent


def _extract_step_info(message) -> dict | None:
    """Convert a LangGraph message into the (tool_used, input, output, tokens) shape.

    Returns None for messages that don't represent a tool call or tool result
    (e.g. the agent's intermediate reasoning text with no tool use).
    """
    msg_type = getattr(message, "type", None)

    if msg_type == "ai" and getattr(message, "tool_calls", None):
        # one AI message can contain multiple tool calls; we handle the common
        # case of one tool call per step, which matches our 3-tool design
        call = message.tool_calls[0]
        token_count = 0
        usage = getattr(message, "usage_metadata", None)
        if usage:
            token_count = usage.get("total_tokens", 0)
        return {
            "tool_used": call["name"],
            "input_summary": call["args"],
            "output_summary": None,  # filled in when the matching ToolMessage arrives
            "token_count": token_count,
        }

    if msg_type == "tool":
        return {
            "tool_used": getattr(message, "name", "unknown_tool"),
            "input_summary": None,  # already captured by the preceding AI message
            "output_summary": str(message.content)[:500],
            "token_count": 0,
        }

    return None


def _is_success_loop(anomaly: dict) -> bool:
    """True if an infinite_loop anomaly's repeating cycle includes a
    successful run_code call — meaning the agent already finished the task
    correctly and is just redundantly re-verifying, not stuck on a real
    failure.

    The normal rollback response ("roll back and try a different approach")
    is counterproductive here, since there's nothing wrong to fix —
    discarding a correct, already-passing solution just wastes tokens (see
    tests/measure_rollback_behavior.py). This case is handled separately in
    run_agent(): the run ends as a success instead of rolling back.
    """
    if anomaly.get("type") != "infinite_loop":
        return False
    for tool, _, output in anomaly.get("repeated_signature", ()):
        if tool == "run_code" and not str(output).lower().startswith("error"):
            return True
    return False


def run_agent(task: str) -> dict:
    """Run the coding agent on a task, with full trajectory monitoring.

    Validates the task, runs the agent step by step (rather than as one
    opaque call), logging and checking every step for anomalies, triggering
    rollback when severity crosses threshold, and verifying completion
    against real logged data rather than the agent's self-report.

    Args:
        task: A natural language description of the coding task.

    Returns:
        A dict describing the final outcome: status, message, warning (if
        any), and the full trajectory log for inspection or display.
    """
    validation = validate_task(task)
    if not validation["valid"]:
        return {
            "status": "rejected",
            "message": validation["reason"],
            "trajectory": [],
            "anomalies_by_step": {},
            "rollback_history": [],
        }

    logger = TrajectoryLogger()
    cost_tracker = CostTracker()
    rollback_manager = RollbackManager()

    agent = build_agent()
    messages = [{"role": "user", "content": task}]
    rollback_status = None
    final_ai_message = ""
    pending_tool_call = None  # holds tool_used/input until the matching ToolMessage arrives
    # keyed by step_number; only populated for steps where a detector actually
    # fired, so the dashboard can show anomaly detail without needing to
    # re-run detection itself
    anomalies_by_step = {}

    total_steps_taken = 0

    while total_steps_taken < MAX_TOTAL_STEPS:
        produced_any_step = False

        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            info = _extract_step_info(latest_message)

            if info is None:
                if getattr(latest_message, "type", None) == "ai":
                    final_ai_message = latest_message.content
                continue

            if info["output_summary"] is None:
                # this is the AI's tool-call announcement; hold onto it until
                # the corresponding ToolMessage arrives with the real output
                pending_tool_call = info
                continue

            # this is a ToolMessage; merge it with the pending call info
            if pending_tool_call:
                merged = {
                    "tool_used": pending_tool_call["tool_used"],
                    "input_summary": pending_tool_call["input_summary"],
                    "output_summary": info["output_summary"],
                    "token_count": pending_tool_call["token_count"],
                }
                pending_tool_call = None
            else:
                merged = info

            entry = logger.log_step(
                tool_used=merged["tool_used"],
                input_summary=merged["input_summary"],
                output_summary=merged["output_summary"],
                token_count=merged["token_count"],
            )
            cost_tracker.log_step(merged["token_count"])
            total_steps_taken += 1
            produced_any_step = True

            active_trajectory = logger.get_active_trajectory()

            # first-step file-existence check
            if len(active_trajectory) == 1:
                first_step_check = check_first_step_validity(entry)
                if first_step_check["status"] == "halted":
                    return {
                        "status": "rejected",
                        "message": first_step_check["reason"],
                        "trajectory": active_trajectory,
                        "anomalies_by_step": {},
                        "rollback_history": [],
                    }

            prior_trajectory = active_trajectory[:-1]
            anomalies = run_all_detectors(task, prior_trajectory, entry)

            if anomalies:
                anomalies_by_step[entry["step_number"]] = anomalies

                # A loop whose repeating cycle includes a passing run_code
                # call means the task is already done — the agent is just
                # redundantly re-verifying, not stuck. Trim the trajectory
                # back to just after the first successful confirmation and
                # end the run as a success, instead of rolling back.
                success_loop = next((a for a in anomalies if _is_success_loop(a)), None)
                if success_loop:
                    cycle_length = success_loop.get("cycle_length", 1)
                    repetition_count = success_loop.get("repetition_count", 1)
                    loop_span = cycle_length * repetition_count
                    trim_to = max(0, entry["step_number"] - loop_span + cycle_length)

                    discarded_before = [s for s in logger.get_full_log() if not s["is_active"]]
                    logger.start_new_branch(trim_to)
                    discarded_after = [s for s in logger.get_full_log() if not s["is_active"]]
                    newly_discarded = discarded_after[len(discarded_before):]
                    if newly_discarded:
                        cost_tracker.log_trim(newly_discarded)

                    final_ai_message = (
                        final_ai_message
                        or "Task completed successfully. (Automatically stopped after "
                           "detecting redundant re-verification of an already-passing result.)"
                    )
                    produced_any_step = False  # signals the outer while loop to stop, not retry
                    break

                severity = calculate_severity(anomalies)
                if severity >= ROLLBACK_SEVERITY_THRESHOLD:
                    discarded_before = [s for s in logger.get_full_log() if not s["is_active"]]

                    rollback_result = rollback_manager.attempt_rollback(
                        anomalies=anomalies,
                        severity=severity,
                        trajectory=active_trajectory,
                        original_goal=task,
                        logger=logger,
                    )

                    discarded_after = [s for s in logger.get_full_log() if not s["is_active"]]
                    newly_discarded = discarded_after[len(discarded_before):]
                    if newly_discarded:
                        cost_tracker.log_rollback(newly_discarded)

                    if rollback_result["status"] == "escalated":
                        rollback_status = "escalated"
                        break

                    # resume with the corrective instruction injected
                    messages = [
                        {"role": "user", "content": task},
                        {"role": "user", "content": rollback_result["correction_message"]},
                    ]
                    break  # restart the stream loop with the new message history

        if rollback_status == "escalated":
            break
        if not produced_any_step:
            break  # the agent finished naturally with no new tool call

    final_trajectory = logger.get_active_trajectory()
    result = finalize_run(final_trajectory, final_ai_message, rollback_status)
    result["trajectory"] = final_trajectory
    result["cost_summary"] = cost_tracker.summary()
    result["anomalies_by_step"] = anomalies_by_step
    result["rollback_history"] = rollback_manager.rollback_history
    return result


if __name__ == "__main__":
    task = "Read the file tasks/buggy_add.py, find the bug, fix it, and run it to confirm it works."
    result = run_agent(task)

    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    if result.get("warning"):
        print(f"Warning: {result['warning']}")
    print(f"\nCost summary: {result.get('cost_summary')}")
    print(f"\nTrajectory ({len(result['trajectory'])} steps):")
    for step in result["trajectory"]:
        print(f"  [{step['step_number']}] {step['tool_used']}: {str(step['input_summary'])[:80]}")