"""
Measures how loops actually manifest in real agent trajectories.

Runs three kinds of experiments:

1. CLEAN RUN (buggy_add) — a task the agent can legitimately solve.
   Max repeats should stay low, confirming no false-positive risk.

2. CLEVER STUCK RUN (impossible_fix) — an assertion that seems impossible
   but Gemini can work around (e.g. hardcoding). Tests whether the agent
   escapes before looping.

3. TRULY STUCK RUN (truly_stuck) — imports from a file that doesn't exist,
   producing the exact same ModuleNotFoundError every single run_code call.
   The agent cannot escape this loop regardless of what it writes to the
   main file. This is the real loop detection test.

Run with:
    python3 -m tests.measure_loop_baseline

Requires GOOGLE_API_KEY in .env.
Results saved to tests/fixtures/loop_baseline_results.json.
"""

import json
import time
from dotenv import load_dotenv
from monitor.logger import TrajectoryLogger

load_dotenv()

MAX_STEPS = 12


def extract_signature(step: dict) -> tuple:
    return (step["tool_used"], str(step["input_summary"]), str(step["output_summary"]))


def find_max_consecutive_repeats(trajectory: list) -> int:
    if not trajectory:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(trajectory)):
        if extract_signature(trajectory[i]) == extract_signature(trajectory[i-1]):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def find_first_loop_at(trajectory: list, threshold: int):
    if len(trajectory) < threshold:
        return None
    current_run = 1
    for i in range(1, len(trajectory)):
        if extract_signature(trajectory[i]) == extract_signature(trajectory[i-1]):
            current_run += 1
            if current_run >= threshold:
                return trajectory[i]["step_number"]
        else:
            current_run = 1
    return None


def run_task_and_measure(task: str, label: str) -> dict:
    from agent.agent import build_agent

    print(f"\n{'='*55}")
    print(f"Running: {label}")
    print(f"Task: {task[:70]}")

    logger = TrajectoryLogger()
    agent = build_agent()
    messages = [{"role": "user", "content": task}]
    pending_tool_call = None
    steps_taken = 0

    try:
        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            if steps_taken >= MAX_STEPS:
                print(f"  Hit MAX_STEPS cap ({MAX_STEPS}), stopping.")
                break

            latest = chunk["messages"][-1]
            msg_type = getattr(latest, "type", None)

            if msg_type == "ai" and getattr(latest, "tool_calls", None):
                call = latest.tool_calls[0]
                usage = getattr(latest, "usage_metadata", None)
                token_count = usage.get("total_tokens", 0) if usage else 0
                pending_tool_call = {
                    "tool_used": call["name"],
                    "input_summary": call["args"],
                    "token_count": token_count,
                }
            elif msg_type == "tool" and pending_tool_call:
                merged = {**pending_tool_call,
                          "output_summary": str(latest.content)[:500]}
                logger.log_step(
                    tool_used=merged["tool_used"],
                    input_summary=merged["input_summary"],
                    output_summary=merged["output_summary"],
                    token_count=merged["token_count"],
                )
                steps_taken += 1
                pending_tool_call = None

                traj = logger.get_active_trajectory()
                last = traj[-1]
                print(f"  Step {last['step_number']}: {last['tool_used']} "
                      f"| out: {str(last['output_summary'])[:70]}")

        trajectory = logger.get_active_trajectory()
        max_repeats = find_max_consecutive_repeats(trajectory)

        threshold_detection = {}
        for t in [2, 3, 4, 5]:
            fired_at = find_first_loop_at(trajectory, t)
            threshold_detection[f"threshold_{t}_fires_at_step"] = fired_at

        print(f"\n  Total steps: {len(trajectory)}")
        print(f"  Max consecutive identical repeats: {max_repeats}")
        for t in [2, 3, 4, 5]:
            fired = threshold_detection[f"threshold_{t}_fires_at_step"]
            print(f"  Threshold={t} would fire at: "
                  f"step {fired if fired else 'never'}")

        return {
            "label": label,
            "task": task,
            "steps": len(trajectory),
            "max_consecutive_repeats": max_repeats,
            **threshold_detection,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"label": label, "task": task, "error": str(e)}


if __name__ == "__main__":
    tasks = [
        {
            "task": (
                "Read the file tasks/buggy_add.py, find the bug, "
                "fix it, and run it to confirm it works."
            ),
            "label": "clean_buggy_add",
        },
        {
            "task": (
                "Read the file tasks/truly_stuck.py. Fix the get_value() "
                "function so the assertion passes when you run it. "
                "Run the file after each change to verify."
            ),
            "label": "stuck_truly_stuck",
        },
    ]

    results = []
    for i, t in enumerate(tasks):
        result = run_task_and_measure(t["task"], t["label"])
        results.append(result)
        if i < len(tasks) - 1:
            print("\nWaiting 15s before next run to avoid rate limits...")
            time.sleep(15)

    output_path = "tests/fixtures/loop_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"{'='*55}")
    for r in results:
        if "error" not in r:
            print(f"\n{r['label']}:")
            print(f"  Steps: {r['steps']}, "
                  f"Max repeats: {r['max_consecutive_repeats']}")
            for t in [2, 3, 4, 5]:
                fired = r.get(f"threshold_{t}_fires_at_step")
                print(f"  Threshold={t}: fires at step "
                      f"{fired if fired else 'never'}")
        else:
            print(f"\n{r['label']}: ERROR — {r['error'][:80]}")

    print(f"\nFull results saved to {output_path}")
    print(f"\nKEY QUESTIONS:")
    print(f"  Clean max_repeats < 3 → no false positives at threshold=3")
    print(f"  Stuck threshold_3_fires_at_step → detection latency at threshold=3")