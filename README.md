# Pathlayer

> Runtime guardrails for AI agents. Open source, local-first, no LLM in the detection loop.

Pathlayer sits between an agent and its environment and detects loops, repeated
failures, lack of progress and cost explosions, then **intervenes during
execution**. Most agent failures are execution-control failures, not reasoning
failures.

```
Agent  ->  Pathlayer  ->  LLM + Tools + APIs
```

> Status: early scaffolding. See [`cursor.md`](cursor.md) for the full project
> context, strategy, and V1 architecture.

## Install

```bash
pip install pathlayer  # not published yet
```

## Layout

```
pathlayer/
  adapters/    # capture layer (OpenClaw first; framework-agnostic)
  detectors/   # extraction, signals, CUSUM
  policies/    # arbitration + graded actions
traces/        # labeled corpus for validation
examples/      # before/after demos
tests/
docs/
```

## License

Apache-2.0.
