# Traceback

> A domain-agnostic framework for real-time trajectory anomaly detection and recovery in agentic AI systems.

---

## What is Traceback?

AI agents operate by executing a sequence of decisions and actions — called a **trajectory**. When something goes wrong mid-execution (goal drift, infinite loops, erratic token usage), the agent has no native mechanism to detect or correct the failure. Errors compound silently, and the problem only surfaces at the end — by which point significant compute and tokens have been wasted.

**Traceback** is a two-part system that wraps any AI agent and monitors its trajectory in real time:

1. **Detects anomalies** at the step level as they occur
2. **Pinpoints the exact step** where execution degraded
3. **Automatically rolls back** to the last clean state
4. **Retries with corrective guidance** injected into the agent's context

The MVP validates the framework on a **coding agent** — chosen because code execution provides objective, binary pass/fail evaluation.

---

## System Architecture

```
User submits task
       │
       ▼
Coding Agent (LangGraph + Claude)
       │
       ▼ (each step)
Trajectory Logger ── records step + full state snapshot
       │
       ▼
Anomaly Detector
  ├── Goal Drift       (cosine similarity < threshold)
  ├── Infinite Loop    (repeated tool+input ≥ threshold)
  └── Token Explosion  (token spike > multiplier × rolling avg)
       │
  ┌────┴────┐
  │         │
No anomaly  Anomaly detected
  │         │
  │         ▼
  │    Severity Scorer (0.0 – 1.0)
  │         │
  │         ▼
  │    Rollback Manager
  │      1. Identify last clean step
  │      2. Reconstruct agent state from snapshot
  │      3. Resume from clean state
  │      4. Inject corrective instruction
  │         │
  └────►    ▼
       Continue execution
             │
             ▼
     Streamlit Dashboard (live)
```

---

## Project Structure

```
traceback/
├── agent/                  # The coding agent (LangGraph + Claude)
│   ├── __init__.py
│   ├── agent.py            # LangGraph agent definition
│   └── tools.py            # read_file, write_file, run_code
│
├── monitor/                # Traceback monitoring system
│   ├── __init__.py
│   ├── logger.py           # Trajectory logger + state snapshots
│   ├── detector.py         # Anomaly detector (drift, loop, token)
│   ├── scorer.py           # Severity scorer
│   └── rollback.py         # Rollback and retry manager
│
├── dashboard/              # Streamlit dashboard
│   ├── __init__.py
│   └── app.py
│
├── config/                 # Domain configuration
│   └── domains.py          # Thresholds per domain
│
├── tasks/                  # Hand-crafted buggy Python files for testing
│
├── tests/                  # Unit and integration tests
│   ├── test_agent.py
│   ├── test_detector.py
│   └── test_rollback.py
│
├── docs/                   # Extended documentation
│   └── milestones.md       # Build log per milestone
│
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent framework | LangGraph |
| LLM | Anthropic Claude API (`claude-sonnet-4-20250514`) |
| Semantic similarity | sentence-transformers (cosine similarity) |
| Dashboard | Streamlit |
| State persistence | JSON files |

---

## Anomaly Types

| Type | Description | Default Threshold |
|---|---|---|
| **Goal Drift** | Agent's actions become semantically irrelevant to the original goal | Cosine similarity < 0.4 |
| **Infinite Loop** | Same tool + input combination repeats consecutively | ≥ 3 repetitions |
| **Token Explosion** | Single step token usage spikes above rolling average | > 3.0× rolling avg |

All thresholds are configurable per domain in `config/domains.py`.

---

## Rollback Strategies

| Strategy | Use Case |
|---|---|
| Roll back 1 step | Minor drift |
| Roll back to last clean step | Cascading errors |
| Full restart with correction | Severe state corruption |

---

## Research Claims

1. **Domain-agnostic detection** — semantic, statistical, and relative measures that adapt to any agent without redesigning detection logic
2. **Threshold portability** — domain-specific tuning requires only recalibrating config values, not architectural changes
3. **State reconstruction is feasible** — full agent state snapshots saved at every step are sufficient to reconstruct and resume from any prior point
4. **Early detection saves cost** — catching anomalies at the step level significantly reduces wasted tokens vs. silent failure and full restarts

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/traceback.git
cd traceback

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your Anthropic API key to .env

# 5. Run the dashboard
streamlit run dashboard/app.py
```

---

## Positioning

Traceback is distinct from adjacent work:

- **TrajectoryGuard** — strong detector but requires training data and has no recovery mechanism. Traceback is training-free with a full detect → rollback → retry loop.
- **AgentDoG** — focuses on safety/alignment guardrailing. Traceback focuses on execution reliability.
- **WebRollback** — agent self-decides to roll back. Traceback uses an external observer that detects *why* rollback is needed and injects corrective guidance.

The central gap Traceback fills: **training-free detection + automated state rollback + corrective retry in a single, domain-agnostic framework.**
