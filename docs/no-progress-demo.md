# The agent that keeps busy without moving forward

This page explains, without jargon, what the `demo_no_progress.py` demo shows
and how to run it. It is written to be shared as is, including with people who
do not code: you can read "What it proves" and the output below without
installing anything.

---

## In one sentence

An AI agent can look busy -- doing something different at every turn -- while
never actually moving forward. Because it does not repeat itself, classic
guardrails see nothing. Pathlayer notices and steps in gradually: it **warns**,
then **forces a progress summary**, then **stops** the agent.

Runaway agents have a real cost. In one publicly reported case, two agents in a
production pipeline kept exchanging messages for 11 days, and the bill reached
about $47,000 ([the team's account](https://pub.towardsai.net/we-spent-47-000-running-ai-agents-in-production-heres-what-nobody-tells-you-about-a2a-and-mcp-5f845848de33)).
This demo reproduces the "busy but stuck" pattern on a scripted agent; it does
not replay that team's system.

---

## What it proves

- **Pathlayer catches a problem that loop detection cannot see.** The agent
  changes its action at every turn; only measuring *real* progress (tests that
  pass) reveals that it is stuck.
- **The intervention is gradual, not abrupt.** The agent is not cut off right
  away: it is warned, then asked for a progress summary, and stopped only if
  nothing moves.
- **No false positive.** An agent that really makes progress (same kind of
  activity, but more and more tests pass) is **never** disturbed: it finishes
  on its own. Stopping an agent that works well would be worse than having no
  guardrail at all.

Everything runs **locally, with no AI in the detection loop and no network**:
it is deterministic and reproducible.

---

## The output you get

```text
================================================================
PATHLAYER DEMO - a no-progress slow burn detected and stopped
================================================================

--- SLOW BURN: busy agent, zero progress (the differentiated case) ---
Without Pathlayer: 200+ actions and still going (demo cap 200).
With Pathlayer:    STOP (run drained).
   Repetition detectors : loop=0.00  retry=0.00 (silent)
   No progress detected : no_progress=1.00 (rises and saturates)
   First intervention   : action 6
   Graded ladder        : WARN -> SUMMARIZE -> STOP
   Forced summary       : yes (pathlayer_progress_summary)
   STOP reason          : no_progress: 11 steps without new progress (cusum)
   Work avoided         : ~186 of 200 actions (14 processed).

--- HEALTHY: same activity, but the tests improve (counterexample) ---
With Pathlayer:    8 actions, no_progress=0.10, no intervention. The agent finishes on its own.
```

How to read it:

- **`loop=0.00 retry=0.00`**: the repetition detectors stay silent -- as
  expected, since the agent never repeats itself. That is the whole point of
  this case.
- **`no_progress=1.00`**: the progress detector is certain that nothing moves.
- **`WARN -> SUMMARIZE -> STOP`**: the three intervention steps, in order.
- **`~186 of 200` avoided**: the work Pathlayer saved.
- The **HEALTHY** block shows the counterexample: no intervention.

---

## How to run it yourself

You need Python 3.10 or later.

```bash
# 1. Get the project
git clone https://github.com/PathlayerLabs/pathlayer.git
cd pathlayer

# 2. Create an isolated environment and install Pathlayer (+ LangGraph support)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[langgraph]"

# 3. Run the demo
python examples/demo_no_progress.py
```

The demo finishes in a few seconds and prints exactly the output above.

---

## What the demo does NOT claim

It shows an agent that produces a **measurable progress signal** (here, test
results). It does not claim to catch a stuck agent that has *no* objective
marker at all (no tests, no code tracking, no exit code). For that harder case,
an operator can declare what progress means with a progress-proof contract; see
[the corpus measurement](progress-proof-corpus-measurement.md) for what that
covers and what it does not.

---

## Going further (technical)

The architecture (LangGraph adapter, `advisory` signals, CUSUM, full runtime
path) is documented in [`examples/README.md`](../examples/README.md), section
`demo_no_progress.py`.
