"""External deterministic probe used only by the ANC-3 corpus benchmark.

The scenario file contains a human-authored event index for each checkpoint.
This process reads only that file and its own invocation counter: it never
inspects an agent event, tool name, result, or free-form text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("counter", type=Path)
    args = parser.parse_args()

    raw = json.loads(args.scenario.read_text(encoding="utf-8"))
    checkpoints = raw.get("checkpoints") if isinstance(raw, dict) else None
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("scenario checkpoints must be a non-empty object")

    try:
        index = int(args.counter.read_text(encoding="utf-8"))
    except FileNotFoundError:
        index = 0
    args.counter.write_text(str(index + 1), encoding="utf-8")

    result: dict[str, bool] = {}
    for checkpoint_id, true_at in checkpoints.items():
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint ids must be non-empty strings")
        if true_at is not None and (
            not isinstance(true_at, int) or isinstance(true_at, bool) or true_at < 0
        ):
            raise ValueError("checkpoint true_at must be a non-negative integer or null")
        result[checkpoint_id] = true_at is not None and index >= true_at

    json.dump({"checkpoints": result}, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
