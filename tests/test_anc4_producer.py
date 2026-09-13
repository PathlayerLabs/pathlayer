"""ANC-4: the ANC-2 producer makes collection failures explicit, never fatal.

A probe failure still degrades to "no proof this tick" (ANC-2's fail-soft
contract, unchanged), but it is now logged with a fixed category instead of
disappearing silently, and a proof that omits required checkpoints says so.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path

import pytest

from pathlayer.config import PathlayerConfig
from pathlayer.detectors.proof_producer import (
    ManifestError,
    ProgressProofProducer,
    build_producer,
    producer_from_config,
)

_LOGGER = "pathlayer.detectors.proof_producer"


def _manifest(tmp_path: Path, required: tuple[str, ...] = ("built", "tested")) -> Path:
    path = tmp_path / "proof_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task-1",
                "proof_id": "proof-1",
                "scope_id": "scope-1",
                "kind": "checkpoint_set",
                "required": list(required),
            }
        ),
        encoding="utf-8",
    )
    return path


def _producer(tmp_path: Path, body: str, *, timeout: float = 10.0) -> ProgressProofProducer:
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return build_producer(str(_manifest(tmp_path)), (sys.executable, str(script)), timeout)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("body", "category"),
    [
        ("import sys; sys.stderr.write('TOKEN=hunter2'); sys.exit(3)", "exited with status 3"),
        ("import time; time.sleep(30)", "timed out"),
        ("print('not json')", "is not JSON"),
        ("import sys; sys.stdout.buffer.write(b'\\xff\\xfe')", "not UTF-8"),
        ("print('[1, 2]')", "not the checkpoints JSON contract"),
        ("import json; print(json.dumps({'checkpoints': ['built']}))", "checkpoints JSON contract"),
        (
            "import json; print(json.dumps({'checkpoints': {'built': 1}}))",
            "checkpoints JSON contract",
        ),
    ],
)
def test_probe_failure_is_inert_and_logged_by_category(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, body: str, category: str
) -> None:
    producer = _producer(tmp_path, body, timeout=1.0)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        payloads = producer.probe()

    messages = _warnings(caplog)
    assert len(messages) == 1
    assert category in messages[0]
    assert "no proof this tick" in messages[0]
    assert "hunter2" not in messages[0]  # probe output is never echoed
    # The same category travels with the event, as the marker's reason.
    assert set(payloads) == {"proof_spec", "progress_proof_unobserved"}
    marker = payloads["progress_proof_unobserved"]
    assert marker["unobserved"] == ["built", "tested"]
    assert category in marker["reason"]
    assert "hunter2" not in marker["reason"]


def test_unlaunchable_probe_is_inert_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    producer = build_producer(str(_manifest(tmp_path)), ("pathlayer-no-such-probe-xyz",), 5.0)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        payloads = producer.probe()

    assert any("could not be launched" in message for message in _warnings(caplog))
    assert payloads["progress_proof_unobserved"]["reason"] == "could not be launched"


def test_deeply_nested_output_is_inert_instead_of_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """json.loads raises RecursionError (not ValueError) on deep nesting."""
    producer = _producer(tmp_path, "import sys; sys.stdout.write('[' * 200000 + ']' * 200000)")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        payloads = producer.probe()

    assert any("is not JSON" in message for message in _warnings(caplog))
    assert "progress_proof" not in payloads


def test_omitted_required_checkpoints_are_logged_and_marked_but_the_proof_is_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    producer = _producer(
        tmp_path, "import json; print(json.dumps({'checkpoints': {'built': True}}))"
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        payloads = producer.probe()

    assert payloads["progress_proof"]["checkpoints"] == {"built": True}
    assert payloads["progress_proof"]["sequence"] == 1
    assert payloads["progress_proof_unobserved"] == {
        "spec_hash": producer.spec_hash,
        "unobserved": ["tested"],
        "reason": "did not report 1 of 2 required checkpoint(s)",
    }
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "1 of 2 required checkpoint(s)" in messages[0]
    assert "tested" in messages[0]


def test_empty_report_is_a_total_outage_not_a_malformed_proof(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``{"checkpoints": {}}`` used to become a proof the anchor rejected as
    malformed (empty checkpoints), i.e. a silent freeze."""
    producer = _producer(tmp_path, "import json; print(json.dumps({'checkpoints': {}}))")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        payloads = producer.probe()

    assert set(payloads) == {"proof_spec", "progress_proof_unobserved"}
    assert payloads["progress_proof_unobserved"]["unobserved"] == ["built", "tested"]
    assert "reported none of the required checkpoints" in _warnings(caplog)[0]
    assert producer.probe()["progress_proof_unobserved"] is not None
    # A sequence number is only consumed by a proof.
    ok = _producer(
        tmp_path,
        "import json; print(json.dumps({'checkpoints': {'built': True, 'tested': False}}))",
    )
    assert ok.probe()["progress_proof"]["sequence"] == 1


def test_partial_then_total_outage_end_to_end_never_stops(tmp_path: Path) -> None:
    """The mid-run outage through the real producer and the real detector: a
    productive agent whose probe stops observing its tests is left alone."""
    from pathlayer.detectors.no_progress import NoProgressDetector
    from pathlayer.detectors.proof import ProofProgressAnchor
    from pathlayer.events import ToolEvent

    counter = tmp_path / "count"
    producer = _producer(
        tmp_path,
        f"""
        import json, sys
        path = {str(counter)!r}
        try:
            n = int(open(path).read())
        except FileNotFoundError:
            n = 0
        open(path, "w").write(str(n + 1))
        if n < 2:  # observed, one checkpoint green
            print(json.dumps({{"checkpoints": {{"built": True, "tested": False}}}}))
        elif n < 8:  # tests unobservable, file still observed
            print(json.dumps({{"checkpoints": {{"built": True}}}}))
        else:  # whole probe down
            sys.exit(4)
        """,
    )
    detector = NoProgressDetector([ProofProgressAnchor()])
    state = detector.create_state()
    confidences = []
    for index in range(20):
        event = ToolEvent(tool="edit", args={"i": index})
        producer.attach(event)
        state, signal = detector.step(state, event)
        confidences.append(signal.confidence)
    # Two observed ticks: a baseline, then one flat step (0.1). Eighteen
    # unobservable ticks: not one of them adds anything.
    assert confidences[:2] == [0.0, pytest.approx(0.1)]
    assert confidences[2:] == [0.0] * 18


def test_complete_proof_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    producer = _producer(
        tmp_path,
        "import json; print(json.dumps({'checkpoints': {'built': True, 'tested': False}}))",
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert producer.probe() is not None

    assert _warnings(caplog) == []


def test_producer_from_config_follows_the_progress_proof_section(tmp_path: Path) -> None:
    assert producer_from_config(PathlayerConfig()) is None

    manifest = _manifest(tmp_path)
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest),
                    "probe_command": [sys.executable, "probe.py"],
                    "timeout_seconds": 7.5,
                }
            }
        }
    )
    producer = producer_from_config(config)

    assert producer is not None
    assert producer.spec.required == ("built", "tested")
    assert producer.probe_command == (sys.executable, "probe.py")
    assert producer.timeout_seconds == 7.5

    manifest.write_text("not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        producer_from_config(config)


def test_production_pipeline_never_stops_on_an_outage_after_the_objective(tmp_path: Path) -> None:
    """build_pipeline wires the coding anchors next to the proof anchor. A
    completed agent keeps running a failing command, then its probe dies:
    the proof anchor's outage must not leave the flat exit-code anchor to
    decide alone."""
    from pathlayer.config import build_pipeline
    from pathlayer.events import ToolEvent
    from pathlayer.policies import Action

    counter = tmp_path / "count"
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            path = {str(counter)!r}
            try:
                n = int(open(path).read())
            except FileNotFoundError:
                n = 0
            open(path, "w").write(str(n + 1))
            if n >= 15:
                sys.exit(4)
            print(json.dumps({{"checkpoints": {{"built": True, "tested": True}}}}))
            """
        ),
        encoding="utf-8",
    )
    section = {
        "enabled": True,
        "manifest_path": str(_manifest(tmp_path)),
        "probe_command": [sys.executable, str(script)],
    }
    config = PathlayerConfig.from_mapping({"detectors": {"progress_proof": section}})
    producer = producer_from_config(config)
    assert producer is not None
    with_contract = build_pipeline(config)
    without_contract = build_pipeline(PathlayerConfig())
    for index in range(30):
        event = ToolEvent(tool="shell", args={"i": index}, result={"exit_code": 1})
        producer.attach(event)
        with_contract.process(event)
        without_contract.process(event)
        decision = with_contract.last_decision
        assert decision is None or decision.arbitrated_action is not Action.STOP, index
    assert "progress_proof_unobserved" in event.metadata
    # Without a contract nothing tells Pathlayer this agent is done: the flat
    # exit codes stop it (zero-config behaviour, unchanged).
    assert without_contract.stopped
