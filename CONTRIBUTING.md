# Contributing to Pathlayer

Pathlayer detects agent execution failures deterministically: no LLM, network
call, randomness, or wall-clock read sits in the detection loop. Contributions
keep that property and keep false positives on healthy runs as the first risk
to rule out.

## Workflow

- Start from a GitHub issue, and keep one issue per branch and pull request.
- Add or update tests for every behavior change. New thresholds or detection
  logic need labeled traces under `traces/`, including healthy
  counterexamples, and must stay safe on the whole golden corpus.
- An optimized incremental `step()` must match its batch `evaluate(history)`
  specification on every event prefix.
- Keep failure signals and progress evidence separate: normalize a signal's
  confidence where a policy compares or multiplies it, so a NaN or
  out-of-range value cannot propagate, and pass positive progress evidence
  through the policy's separate `evidence` mapping, never through the signal
  sequence.
- For every detector or policy semantic change, decide whether
  `STATE_SEMANTICS_VERSION` in `pathlayer/store.py` must be bumped: bump it when
  unchanged configuration can change future decisions in a way full prefix
  replay cannot expose.
- Documentation states only what is enforced and proven: distinguish observed,
  advisory, and enforced actions.
- Pull requests are squash-merged: one pull request, one commit on `main`.

## Validation

Install the development dependencies in a virtual environment, then the Git
hooks:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` sets up two hooks from `.pre-commit-config.yaml`. Each
commit runs the whitespace, YAML and TOML checks, `ruff check --fix` and
`ruff format`; each push also runs `mypy`, which takes a few seconds. The push
hook uses the `mypy` of the active environment, so push with the virtual
environment activated. Ruff already sorts imports and formats code: do not add
Black or isort, which would fight it. The hooks give early feedback on your
machine; they do not replace CI, which also runs the test suite.

Run the full gate before a pull request is ready for review:

```bash
ruff check .
ruff format --check .
mypy
pytest
pre-commit run --all-files
```

`pytest` enforces branch coverage of at least 85% through `pyproject.toml`.
When packaging or README usage changes, also run the relevant build and smoke
checks from `scripts/` and `.github/workflows/ci.yml`; install their extra
tooling with `python -m pip install uv twine`.

## Fast local iteration loop

Branch coverage makes coverage.py fall back to its slow C tracer. For the
edit-run loop only, `.coveragerc.fast` measures lines with `sys.monitoring`
instead (Python 3.12+), about 3.5x less CPU on the hot path:

```bash
COVERAGE_CORE=sysmon pytest --cov-config=.coveragerc.fast --cov-fail-under=0
```

This run does not verify the 85% branch-coverage floor and is not the gate:
plain `pytest`, which the CI `test` job runs, stays mandatory before a pull
request.

## Reporting problems

Use the issue templates. A false positive or a missed detection has its own
template: it is the most useful report, because false positives on healthy
runs are the first risk Pathlayer guards against. Report security problems
privately, as described in [`SECURITY.md`](SECURITY.md).
