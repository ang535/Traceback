# Traceback

> A domain-agnostic framework for real-time trajectory anomaly detection and recovery in agentic AI systems.

---

## What is Traceback?

AI agents operate by executing a sequence of decisions and actions — called a **trajectory**. When something goes wrong mid-execution (goal drift, infinite loops, erratic token usage), the agent has no native mechanism to detect or correct the failure. Errors compound silently, and the problem only surfaces at the end — by which point significant compute and tokens have been wasted.

**Traceback** wraps an AI agent and monitors its trajectory step by step:

1. **Detects anomalies** at the step level as they occur
2. **Scores severity** and decides whether the trajectory needs correcting
3. **Rolls back** to a clean prior state (one step, the last clean step, or a full restart, depending on severity)
4. **Injects a corrective instruction** specific to what went wrong, and retries
5. **Escalates to the user** instead of retrying, when retrying provably can't help

The MVP validates the framework on a **coding agent** — chosen because code execution provides objective, binary pass/fail evaluation.

---

## System Architecture

```
User submits task
       │
       ▼
Coding Agent (LangGraph + Groq/Gemini)
       │
       ▼ (each step)
Trajectory Logger ── records step + branch id (branches on rollback,
       │              old steps kept but marked inactive, not deleted)
       ▼
Anomaly Detector
  ├── Goal Drift          (cosine similarity < threshold, with a
  │                        deterministic file-match override for tasks
  │                        that explicitly name their target file)
  ├── Wrong Target File   (step's target file doesn't match any file
  │                        named in the task — deterministic, not similarity-based)
  ├── Infinite Loop       (same tool+input cycle repeats ≥ threshold)
  └── Token Explosion     (token spike vs. rolling average, or an
                           absolute ceiling if no rolling average exists yet)
       │
  ┌────┴──────────────────┐
  │                        │
No anomaly        Anomaly detected
  │                        │
  │              Is it a pure read-only repeating loop?
  │               (no write_file / run_code in the cycle)
  │                        │
  │              ┌─── yes ─┴─ no ───┐
  │              ▼                  ▼
  │        Escalate            Is it a repeating loop whose only
  │        immediately         run_code call already passed?
  │        (retrying can't          │
  │        change an unchanged  ┌─ yes ─┴─ no ──┐
  │        read)                ▼               ▼
  │                        Trim redundant   Severity Scorer (0.0–1.0)
  │                        steps, declare        │
  │                        success                ▼
  │                                          ≥ ROLLBACK_SEVERITY_THRESHOLD?
  │                                               │
  │                                          ┌─ no → keep going
  │                                          └─ yes → Rollback Manager
  │                                                     1. Pick strategy per anomaly type
  │                                                     2. Full restart if severity AND
  │                                                        ≥2 distinct anomaly types agree
  │                                                     3. Branch from the rollback point
  │                                                     4. Inject a correction message
  │                                                        specific to the anomaly type
  │                                                     5. Escalate if MAX_ROLLBACKS_PER_TASK
  │                                                        is exceeded
  └──────────────────────►    ▼
                         Continue execution
                               │
                               ▼
                  Verified completion check
                  (re-checks the actual last run_code
                  result — doesn't trust the agent's
                  own "I'm done" self-report)
                               │
                               ▼
                     Streamlit Dashboard
                (launch runs, browse history, inspect
                 anomalies and rollback correction messages)
```

---

## Project Structure

```
traceback/
├── agent/
│   ├── agent.py             # LangGraph agent loop: detect → score → rollback/escalate → verify
│   └── tools.py              # read_file, write_file, run_code
│
├── monitor/
│   ├── logger.py              # TrajectoryLogger: step entries, branching on rollback
│   ├── detector.py            # Anomaly detectors: drift, wrong-target-file, loop, token
│   ├── embeddings.py          # Shared sentence-transformers model (all-MiniLM-L6-v2)
│   ├── scorer.py              # Per-anomaly severity scoring + combined severity
│   ├── rollback.py            # Rollback strategy, correction messages, retry budget
│   ├── cost_tracker.py        # Token accounting: useful vs. wasted-on-rollback vs. trimmed
│   ├── completion_check.py    # Verifies completion from real run_code output, not self-report
│   ├── validator.py           # Rejects obviously invalid tasks before running the agent
│   └── run_store.py           # Persists run_agent() results to runs/*.json for the dashboard
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard: launch runs + browse history
│
├── tasks/                     # Hand-crafted buggy Python files used as test fixtures
│
├── tests/                     # Offline tuning/validation scripts + labeled scenario fixtures
│   ├── fixtures/               # Labeled scenario sets used to tune thresholds against F1
│   ├── measure_*.py            # Collects real data from live runs (token baselines, etc.)
│   ├── tune_*.py                # Sweeps a constant against labeled scenarios, reports F1
│   └── verify_*.py              # Deterministic checks that don't require a live model call
│
├── docs/                      # Build log: one markdown file per fix/feature, with the
│                               #   reasoning and verification behind it (not tracked in git)
│
├── runs/                      # Saved run JSON files (gitignored)
├── domains.py                 # Per-domain threshold presets — not yet wired into the
│                               #   detection pipeline; monitor/detector.py's thresholds
│                               #   are still hardcoded module constants (see Known Limitations)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent framework | LangGraph (`create_react_agent`) |
| LLM | Groq (`openai/gpt-oss-20b`, default) or Gemini (`gemini-2.5-flash`) — switched via `PROVIDER` in `agent/agent.py` |
| Semantic similarity | sentence-transformers (`all-MiniLM-L6-v2`), cosine similarity |
| Dashboard | Streamlit |
| State persistence | JSON files (`runs/*.json`) |

Groq is the default because its free tier (1000 req/day) supports iterating on tuning scripts without hitting Gemini's 20 req/day free-tier ceiling.

---

## Anomaly Types

| Type | Description | Current threshold |
|---|---|---|
| **Goal Drift** | Agent's action becomes semantically irrelevant to the task | Cosine similarity < 0.6, unless the step's target file is explicitly named in the task (deterministic override, since embedding similarity alone under-detects "on-task" behavior on long, multi-part tasks) |
| **Wrong Target File** | Step's target file doesn't match any file named in the task | Deterministic exact-match check, not similarity-based |
| **Infinite Loop** | Same tool + input (or short cycle of tools) repeats consecutively | ≥ 3 repetitions, cycle length up to 3 |
| **Token Explosion** | Single step's token usage spikes | > 2.2× rolling average, or an absolute ceiling (4000 tokens) if no rolling average exists yet |

Every threshold above was empirically set — swept against a labeled scenario set (`tests/fixtures/`) and/or measured from real agent runs, with the reasoning and data recorded next to each constant in `monitor/detector.py` and `monitor/scorer.py`. None are default guesses.

### Beyond raw detection

Two behaviors sit on top of the four detectors above, because "an anomaly fired" isn't always "retry is the right move":

- **Success-loop early stop** — if a detected loop's repeating cycle contains a `run_code` call that already passed, the agent isn't stuck, it's just re-verifying finished work. Redundant steps are trimmed and the run is marked successful, instead of rolling back correct code.
- **Unproductive read-loop escalation** — if a detected loop's repeating cycle is *entirely* `read_file` calls (no `write_file`, no `run_code`), retrying can't help: re-reading unchanged content produces no new information, and a correction message telling the model to "try something different" doesn't change what it re-reads. This escalates immediately instead of burning the retry budget on a guaranteed-identical repeat.

---

## Rollback Strategies

| Strategy | Trigger |
|---|---|
| Roll back 1 step | Isolated token_explosion, or mild goal_drift |
| Roll back to last clean step | Infinite loop, or severe goal_drift |
| Full restart | Combined severity ≥ 0.85 **and** ≥ 2 distinct anomaly types firing together (a single anomaly, however extreme, can't trigger this alone — see the two-signal gate in `monitor/rollback.py`) |

Every rollback that isn't escalated injects a **correction message specific to the anomaly type** that triggered it (e.g. "be more concise" for token_explosion, "refocus on: {task}" for goal_drift) before the agent retries. `MAX_ROLLBACKS_PER_TASK` (3) caps how many times this can happen before the run escalates to the user with the generic retry-exhaustion message, or with a specific reason if the cause is known (e.g. the read-loop case above).

Token spend is tracked three ways per run: useful tokens, tokens wasted on genuine rollbacks, and tokens discarded by success-loop trims (which aren't counted as "wasted," since the underlying work was correct) — all visible on the dashboard.

---

## Dashboard

`streamlit run dashboard/app.py` gives you:

- **A task box and Run button**, always visible, to launch a new agent run against the live model
- **Dashboard tab** — the result of the current or most recently loaded run: a run summary (steps, tokens, tokens wasted on rollbacks, anomalies flagged, rollbacks, success-loop trims, each with a tooltip), the full execution trajectory (with rows containing an anomaly highlighted), a per-step token bar chart, expandable raw anomaly detail per flagged step, and expandable rollback history showing the actual correction message sent back to the agent on each attempt
- **History tab** — browse and reload any past run saved to `runs/*.json`

---

## Known Limitations

- **`domains.py` isn't wired in yet.** It defines per-domain threshold presets, but `monitor/detector.py`'s actual thresholds are hardcoded module constants, independently tuned against real coding-agent data. Swapping domains today means manually editing those constants, not passing a domain name.
- **`run_code` only executes a single file** (`python3 <filepath>`) — it doesn't resolve package-relative imports or run a test framework like pytest. Tasks that need `from tasks.module import fn` will fail for reasons unrelated to the agent's actual code; self-contained scripts avoid this.
- **Embedding similarity requires `sentence-transformers`,** which is a heavier dependency than the rest of the stack — goal-drift detection can't run at all without it installed.
- **The dashboard's "live" run isn't streamed.** `run_agent()` is a blocking call that returns one full result at the end, so launching a run shows a spinner, then the complete result — not a step-by-step feed while it runs.

---

## Research Claims

1. **Domain-agnostic detection** — semantic, statistical, and deterministic signals combine into one severity score without requiring training data
2. **State reconstruction is feasible** — branch-tracked trajectory logging (not deletion) is sufficient to roll back to and resume from any prior step
3. **Not every anomaly should trigger the same response** — a repeating loop that already succeeded should stop, not roll back; a repeating loop that can't possibly make progress should escalate, not retry. Both required distinguishing anomaly *type and context*, not just severity
4. **Early detection saves cost** — catching anomalies at the step level, and routing them to the cheapest sufficient response (trim vs. one-step rollback vs. full restart vs. immediate escalation), reduces wasted tokens vs. either silent failure or uniformly retrying everything

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/ang535/Traceback.git
cd Traceback

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY (default provider) and/or GOOGLE_API_KEY (Gemini) to .env

# 5. Run the dashboard
streamlit run dashboard/app.py
```

To switch providers, change `PROVIDER = "groq"` to `PROVIDER = "gemini"` in `agent/agent.py`.

---

## Positioning

Traceback is distinct from adjacent work:

- **TrajectoryGuard** — strong detector but requires training data and has no recovery mechanism. Traceback is training-free with a full detect → rollback → retry loop.
- **AgentDoG** — focuses on safety/alignment guardrailing. Traceback focuses on execution reliability.
- **WebRollback** — agent self-decides to roll back. Traceback uses an external observer that detects *why* rollback is needed and injects corrective guidance.

The central gap Traceback fills: **training-free detection + automated state rollback + corrective retry in a single framework**, with per-anomaly-type strategy selection instead of one uniform rollback response.
