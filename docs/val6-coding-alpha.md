# Coding Alpha demo (VAL-6)

From a source checkout, install `pip install -e '.[dev,langgraph,yaml]'`, then run:

```bash
python examples/demo_coding_alpha.py
```

This is a scripted agent running real LangGraph tools in disposable workspaces,
not an LLM generalization benchmark. Its edit tool writes a calculator module and
its test tool actually executes pytest. A separately compiled ANC-4 contract
checks the objective; the STATE-1 configuration maps edit arguments to files.
No production detector threshold is changed.

The five scenarios distinguish productive repetition, identical stagnant edits,
new file content without successful tests, a missing validation executable
(`collection_failed`), and nested tool boundaries (`unavailable`). Missing evidence
must not create a no-progress signal; independent loop detection can still STOP.

Each run is compared with the same script without proofs and with replay after
the workspace has been deleted. The output reports contract checkpoints, resource
state, calls since progress and the recorded intervention. An observed file change
is not proof that the tests pass. Test/build collection adds subprocess cost and
is appropriate only for bounded, fast checks.

Run the acceptance assertions with:

```bash
pytest tests/test_val6_coding_alpha.py tests/test_val6_diagnostic.py
```

The demo's temporary traces are removed on exit. Call `run_scenario(name, root)`
with a retained root to keep `trace.json` for inspection.
