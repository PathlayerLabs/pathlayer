"""Verify that an index serves exactly the artifacts built by this release."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _local_hashes(dist: Path) -> dict[str, str]:
    artifacts = sorted((*dist.glob("*.whl"), *dist.glob("*.tar.gz")))
    if len(artifacts) != 2:
        raise SystemExit(f"expected two local artifacts, found {len(artifacts)}")
    return {
        artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest() for artifact in artifacts
    }


def _fetch_release(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)  # type: ignore[no-any-return]


def _wait_for_hashes(
    url: str,
    *,
    expected: dict[str, str],
    attempts: int,
    delay: int,
) -> None:
    last_remote: dict[str, str] | None = None
    for attempt in range(1, attempts + 1):
        try:
            release = _fetch_release(url)
            last_remote = {item["filename"]: item["digests"]["sha256"] for item in release["urls"]}
            if last_remote == expected:
                return
        except Exception:
            if attempt == attempts:
                raise
        if attempt < attempts:
            time.sleep(min(delay * (2 ** (attempt - 1)), 60))
    raise SystemExit(f"index artifact hashes differ: remote={last_remote}, local={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--index-api", required=True)
    parser.add_argument("--project", default="pathlayer")
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--delay", type=int, default=5)
    args = parser.parse_args()

    endpoint = f"{args.index_api.rstrip('/')}/{args.project}/{args.version}/json"
    local = _local_hashes(args.dist)
    _wait_for_hashes(
        endpoint,
        expected=local,
        attempts=args.attempts,
        delay=args.delay,
    )
    print(f"index-hashes-ok project={args.project} version={args.version}")


if __name__ == "__main__":
    main()
