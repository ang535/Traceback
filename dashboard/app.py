"""Traceback dashboard.

Two things you can do here: launch a new agent run and see its result, or
browse past runs saved by monitor/run_store.py. Both use the same detail
view underneath, since a run's result looks the same whether it just
finished or finished three days ago.

Run with:
    streamlit run dashboard/app.py

Note on "live": run_agent() is a blocking call — it doesn't hand back
individual steps as they happen, only the final result once the whole run
completes. So launching a run here shows a spinner, then the full result
all at once, not a step-by-step feed while it's running. True step
streaming would need run_agent() itself to be refactored into a generator
(yielding each step as it's logged) instead of returning one dict at the
end — a reasonable next step if watching steps arrive live turns out to
matter more than seeing the finished result quickly.
"""

import os
import sys

# dashboard/ is a sibling of agent/ and monitor/, not a parent of them —
# `streamlit run dashboard/app.py` only puts dashboard/ itself on sys.path,
# so without this, `from agent.agent import run_agent` fails unless
# Streamlit happens to be launched from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from monitor.run_store import save_run, list_runs, load_run

st.set_page_config(page_title="Traceback", layout="wide")


STATUS_DISPLAY = {
    "success": ("Success", st.success),
    "unverified": ("Unverified", st.warning),
    "escalated": ("Escalated", st.error),
    "rejected": ("Rejected", st.error),
}


def format_run_label(summary: dict) -> str:
    import datetime

    when = datetime.datetime.fromtimestamp(summary["saved_at"]).strftime("%b %d, %H:%M")
    task_preview = summary["task"][:50] + ("..." if len(summary["task"]) > 50 else "")
    return f"{when} — {summary['status']} — {task_preview}"


def render_run_detail(record: dict) -> None:
    """Render the full detail view for one run — shared by both the
    just-launched run and anything picked from history, since a saved
    record looks identical either way."""
    status = record.get("status", "unknown")
    label, status_fn = STATUS_DISPLAY.get(status, (status, st.info))

    st.subheader(record.get("task", "(no task recorded)"))
    status_fn(f"**{label}** — {record.get('message', '')}")
    if record.get("warning"):
        st.warning(record["warning"])

    trajectory = record.get("trajectory", [])
    cost_summary = record.get("cost_summary", {}) or {}
    anomalies_by_step = record.get("anomalies_by_step", {}) or {}
    rollback_history = record.get("rollback_history", []) or []

    total_anomalies = sum(len(v) for v in anomalies_by_step.values())

    cols = st.columns(5)
    cols[0].metric("Steps", len(trajectory))
    cols[1].metric("Total tokens", cost_summary.get("total_tokens", 0))
    cols[2].metric("Wasted on rollbacks", cost_summary.get("tokens_wasted_on_rollbacks", 0))
    cols[3].metric("Anomalies flagged", total_anomalies)
    cols[4].metric("Rollbacks", len(rollback_history))

    if not trajectory:
        st.info("No steps were logged for this run.")
        return

    st.markdown("#### Trajectory")

    table_rows = []
    for step in trajectory:
        step_anomalies = anomalies_by_step.get(step["step_number"]) or anomalies_by_step.get(str(step["step_number"])) or []
        table_rows.append({
            "Step": step["step_number"],
            "Tool": step["tool_used"],
            "Input": str(step.get("input_summary", ""))[:80],
            "Output": str(step.get("output_summary", ""))[:80],
            "Tokens": step.get("token_count", 0),
            "Anomalies": ", ".join(a["type"] for a in step_anomalies) if step_anomalies else "",
        })
    st.dataframe(table_rows, width="stretch", hide_index=True)

    st.markdown("#### Tokens per step")
    st.bar_chart({s["step_number"]: s.get("token_count", 0) for s in trajectory})

    if total_anomalies:
        st.markdown("#### Anomaly detail")
        for step in trajectory:
            step_anomalies = anomalies_by_step.get(step["step_number"]) or anomalies_by_step.get(str(step["step_number"])) or []
            if not step_anomalies:
                continue
            with st.expander(f"Step {step['step_number']} — {len(step_anomalies)} anomaly(ies)"):
                for anomaly in step_anomalies:
                    st.json(anomaly)

    if rollback_history:
        st.markdown("#### Rollback history")
        rollback_rows = [
            {
                "Attempt": r["attempt"],
                "Rolled back to step": r["rollback_point"],
                "Anomaly types": ", ".join(r["anomaly_types"]),
                "Severity": round(r["severity"], 3),
            }
            for r in rollback_history
        ]
        st.dataframe(rollback_rows, width="stretch", hide_index=True)


def main():
    st.title("Traceback")
    st.caption("Trajectory anomaly detection and recovery for agentic AI systems")

    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = None
    if "current_run_record" not in st.session_state:
        st.session_state.current_run_record = None

    with st.sidebar:
        st.header("New run")
        try:
            from agent.agent import PROVIDER, GROQ_MODEL, GEMINI_MODEL

            active_model = GROQ_MODEL if PROVIDER == "groq" else GEMINI_MODEL
            st.caption(f"Provider: `{PROVIDER}` · Model: `{active_model}`")
        except Exception:
            pass  # provider info is a nice-to-have, not worth failing the page over

        task = st.text_area(
            "Task",
            placeholder="Read the file tasks/buggy_add.py, find the bug, fix it, and run it to confirm it works.",
            height=100,
        )
        run_clicked = st.button("Run agent", type="primary", disabled=not task.strip())

        st.divider()
        st.header("History")

        runs = list_runs()
        if not runs:
            st.caption("No saved runs yet — run something above.")
        else:
            options = {format_run_label(r): r["run_id"] for r in runs}
            picked_label = st.selectbox("Past runs", list(options.keys()))
            if st.button("Load selected run"):
                st.session_state.current_run_id = options[picked_label]
                st.session_state.current_run_record = load_run(options[picked_label])

    if run_clicked:
        from agent.agent import run_agent

        with st.spinner("Agent running — this calls the real model, may take a while..."):
            try:
                result = run_agent(task)
                run_id = save_run(task, result)
                st.session_state.current_run_id = run_id
                st.session_state.current_run_record = load_run(run_id)
            except Exception as e:
                st.error(f"Run failed: {e}")

    if st.session_state.current_run_record:
        render_run_detail(st.session_state.current_run_record)
    else:
        st.info("Launch a run from the sidebar, or load one from history, to see details here.")


if __name__ == "__main__":
    main()
