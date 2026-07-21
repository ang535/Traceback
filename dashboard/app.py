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

import html
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

    # The task text is already visible in the input box above (or, when
    # browsing History, in that run's picker label) — repeating the full
    # prompt again here was pure duplication.
    status_fn(f"**{label}** — {record.get('message', '')}")
    if record.get("warning"):
        st.warning(record["warning"])

    trajectory = record.get("trajectory", [])
    cost_summary = record.get("cost_summary", {}) or {}
    anomalies_by_step = record.get("anomalies_by_step", {}) or {}
    rollback_history = record.get("rollback_history", []) or []

    total_anomalies = sum(len(v) for v in anomalies_by_step.values())

    st.markdown("#### Run summary")
    cols = st.columns(6)
    cols[0].metric(
        "Steps", len(trajectory),
        help="How many actions the agent took in this run.",
    )
    cols[1].metric(
        "Total tokens", cost_summary.get("total_tokens", 0),
        help="Total tokens used across every step, including any steps that were later discarded.",
    )
    cols[2].metric(
        "Wasted on rollbacks", cost_summary.get("tokens_wasted_on_rollbacks", 0),
        help="Tokens spent on steps that were thrown away when the agent had to roll back and "
             "retry after a real problem was detected. Part of the total above, not extra.",
    )
    cols[3].metric(
        "Anomalies flagged", total_anomalies,
        help="How many times the monitoring system flagged a potential problem, such as a "
             "repeating loop, a sudden spike in response length, or the agent drifting off-task.",
    )
    cols[4].metric(
        "Rollbacks", len(rollback_history),
        help="How many times the agent's recent steps were discarded and retried after a "
             "genuine problem was detected.",
    )
    # trim_count/tokens_wasted_on_trims added when log_trim() was split out
    # from log_rollback() (see docs/step_and_rollback_budget_tuning.md,
    # Round 4) — a success-loop early stop is NOT a rollback, but it still
    # discards steps and is worth seeing on the dashboard rather than
    # disappearing silently now that it's correctly excluded from the
    # Rollbacks count above.
    cols[5].metric(
        "Trims (success-loop)", cost_summary.get("trim_count", 0),
        help="How many times redundant steps were automatically removed after the agent "
             "finished the task correctly but kept re-checking its own work anyway.",
    )

    if not trajectory:
        st.info("No steps were logged for this run.")
        return

    st.markdown("#### Execution Trajectory")
    st.caption("Every action the agent took, in order, with what it did and what came back.")

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

    # st.dataframe doesn't support per-row conditional styling on its own —
    # that needs a pandas Styler, which needs jinja2, an extra dependency
    # not otherwise required anywhere in this project. A small hand-built
    # HTML table avoids that dependency entirely and gives full control over
    # the highlight; the tradeoff is losing st.dataframe's built-in sort/
    # search/fullscreen/download controls for this one table.
    _COLUMNS = ["Step", "Tool", "Input", "Output", "Tokens", "Anomalies"]
    header_html = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.4);">{col}</th>'
        for col in _COLUMNS
    )
    body_rows_html = []
    for row in table_rows:
        row_bg = "background-color: rgba(255, 90, 90, 0.22);" if row["Anomalies"] else ""
        cells_html = "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.15);">{html.escape(str(row[col]))}</td>'
            for col in _COLUMNS
        )
        body_rows_html.append(f'<tr style="{row_bg}">{cells_html}</tr>')

    st.markdown(
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(body_rows_html)}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Tokens per step")
    # st.bar_chart stretches bars to fill the full chart width regardless of
    # how few steps there are, so a short run looks like a couple of thick
    # blocks. A single fixed mark_bar size overcorrects the other way (too
    # thin) once a run only has 2-4 steps. Scale bar width with step count
    # instead: wide for short runs, capped down as steps increase, with a
    # floor so a long (~25-step) run doesn't get slivers either.
    import altair as alt
    import pandas as pd

    bar_size = max(14, min(45, 220 // max(1, len(trajectory))))

    chart_data = pd.DataFrame({
        "Step": [s["step_number"] for s in trajectory],
        "Tokens": [s.get("token_count", 0) for s in trajectory],
    })
    chart = (
        alt.Chart(chart_data)
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X("Step:O", title="Step"),
            y=alt.Y("Tokens:Q", title="Tokens"),
        )
    )
    st.altair_chart(chart, width="stretch")

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
        st.caption("What went wrong on each attempt, and the corrective instruction sent back to the agent.")
        # A plain dataframe can't fit the correction message (a full sentence)
        # into a table cell without wrapping badly, so each attempt gets its
        # own expander instead — summary line matches the old table's
        # columns, correction_message goes in the body where there's room.
        for r in rollback_history:
            summary = (
                f"Attempt {r['attempt']} — rolled back to step {r['rollback_point']} "
                f"— {', '.join(r['anomaly_types'])} (severity {round(r['severity'], 3)})"
            )
            with st.expander(summary):
                st.write(r.get("correction_message", "No correction message recorded for this attempt."))


def main():
    # A plain st.title() + st.caption() pair renders with a visible gap
    # between them (Streamlit's default block spacing), which reads as two
    # unrelated pieces of text rather than a heading + subheading. Wrapping
    # both in one HTML block with the subheading's top margin removed pulls
    # them together. opacity (rather than a hardcoded gray) is used for the
    # subheading color so it still looks right under either theme.
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <h1 style="margin-bottom: 0.15rem;">Traceback</h1>
            <p style="margin-top: 0; opacity: 0.6; font-size: 1rem;">
                Trajectory anomaly detection and recovery for agentic AI systems
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = None
    if "current_run_record" not in st.session_state:
        st.session_state.current_run_record = None

    try:
        from agent.agent import PROVIDER, GROQ_MODEL, GEMINI_MODEL

        active_model = GROQ_MODEL if PROVIDER == "groq" else GEMINI_MODEL
        st.caption(f"Provider: `{PROVIDER}` · Model: `{active_model}`")
    except Exception:
        pass  # provider info is a nice-to-have, not worth failing the page over

    task = st.text_area(
        "Task",
        placeholder="Read the file tasks/buggy_add.py, find the bug, fix it, and run it to confirm it works.",
        height=140,
    )
    run_clicked = st.button("Run agent", type="primary", disabled=not task.strip())

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

    st.divider()

    dashboard_tab, history_tab = st.tabs(["Dashboard", "History"])

    with dashboard_tab:
        if st.session_state.current_run_record:
            render_run_detail(st.session_state.current_run_record)
        else:
            st.info("Launch a run above, or load one from the History tab, to see details here.")

    with history_tab:
        runs = list_runs()
        if not runs:
            st.caption("No saved runs yet — run something above.")
        else:
            options = {format_run_label(r): r["run_id"] for r in runs}
            picked_label = st.selectbox("Past runs", list(options.keys()))
            if st.button("Load selected run"):
                st.session_state.current_run_id = options[picked_label]
                st.session_state.current_run_record = load_run(options[picked_label])
                # The Dashboard tab's content is built earlier in this same
                # script pass (it's rendered above History), so without a
                # forced rerun it would still show the OLD run for this pass
                # — the user would have to click something else before the
                # newly loaded run actually appeared. Rerunning immediately
                # avoids that one-click lag.
                st.session_state.just_loaded_run = True
                st.rerun()

    if st.session_state.pop("just_loaded_run", False):
        st.toast("Run loaded — check the Dashboard tab.")


if __name__ == "__main__":
    main()
