# OpenClaw 2026.7 hook ordering

Synthetic, content-free fixtures reproducing the hook order and field names
reported in the OpenClaw 2026.7.1 live test described in issue #50.
These are not copies of a pilot's private run: identifiers, tool contents and
usage counts are invented. No prompt, private path or credential is included.

`post_run.json` uses `agent_end` then `llm_output`; `in_run.json` moves only
`llm_output` before the end to prove the same spend regardless of arrival order.
The known split costs $0.021 at the pinned Sonnet 4.5 rates.
