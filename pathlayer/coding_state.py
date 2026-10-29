"""The opt-in ``coding`` StateEvidence preset.

ARCH-2 and VAL-5 proved that a before/after observation can improve a Pathlayer
decision, but the integrator had to name every resource up front. This module
selects them instead, from what a tool call already says about itself::

    detectors:
      state_evidence:
        enabled: true
        preset: coding
        workspace: .

The preset never guesses intent. It maps a tool call to resources only through
explicit, documented rules, and abstains everywhere else:

============================  ===========================================
Call                          Observed
============================  ===========================================
A known file-writing tool     the named file(s), plus the configured scope
A text-editor style call      ``path``, plus the configured scope
(``command`` + ``path``)
A tool declared in ``tools``  what that declaration names
A read, an unknown tool       nothing: the call claims no state
A mutating call whose paths   the configured scope alone, never an invented
cannot be resolved            file -- and nothing at all when no scope is
                              configured
============================  ===========================================

The *configured scope* is what the operator declared once: the git workspace,
an explicit pytest selection, an explicit validation command, whichever are
configured. Falling back to it is the one thing the preset does when a mutating
call is ambiguous -- it observes something the operator chose, rather than
attributing a change to a file the arguments did not clearly name. With
``git: false`` and no tests or build declared there is no such scope, so an
ambiguous call is simply not observed.

Three properties this file exists to protect:

- **Attribution.** A resource is observed for a call only when that call's own
  arguments name it, or when the operator configured it as the scope. Paths
  that escape the workspace, are unusable, or are too many to attribute are
  abstentions.
- **Abstention over invention.** Every rejection above yields fewer
  observations, never a fabricated one. The consumer treats a missing
  observation as a gap, never as "unchanged" (see
  :mod:`pathlayer.detectors.state_progress`).
- **An observed effect is not success.** The preset says a file, tree, test
  selection or command changed. Whether the task advanced is what an ANC-4
  contract answers, and the coding anchor here defers to that evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathlayer._pytest_evidence_protocol import MAX_NODEID_LENGTH, MAX_OUTCOMES
from pathlayer.coding_state_collectors import (
    CommandStateSensor,
    GitWorkspaceError,
    GitWorkspaceProbe,
    GitWorkspaceStateSensor,
    git_directory,
    validate_command,
)
from pathlayer.config import ConfigError, PathlayerConfig, StateEvidenceConfig, ToolRuleConfig
from pathlayer.state_collectors import resolve_non_strict
from pathlayer.state_evidence import (
    FilesystemStateSensor,
    PytestStateSensor,
    StateEvidenceProducer,
    StateSensor,
)

CODING_PRESET = "coding"

WRITE = "write"
READ = "read"
WORKSPACE = "workspace"
TOOL_EFFECTS = (WRITE, READ, WORKSPACE)

# Attribution stays believable only while a call names a handful of files. A
# wider call (a bulk formatter, a patch touching a tree) falls back to the
# configured scope instead of claiming a long list of resources.
_MAX_OBSERVED_PATHS = 8
# How many path arguments one tool rule may declare. A separate ceiling from
# the one above: a single argument can hold a list, so the two count different
# things and must be able to move independently.
_MAX_RULE_PATH_ARGUMENTS = 8
_MAX_PATH_LENGTH = 1024
# Fixed, unlike the operator-named build id: the preset supports exactly one
# pytest selection, so there is nothing for an operator to tell apart.
_TESTS_RESOURCE_ID = "selection"

# The file tools LangChain ships (``FileManagementToolkit``), which is what a
# LangGraph coding agent uses unless it defines its own. Names and argument
# names are theirs; a project with different tools declares them in
# ``detectors.state_evidence.tools`` rather than having them guessed here.
BUILTIN_TOOL_RULES: Mapping[str, ToolRuleConfig] = {
    "write_file": ToolRuleConfig(effect=WRITE, paths=("file_path",)),
    "file_delete": ToolRuleConfig(effect=WRITE, paths=("file_path",)),
    "move_file": ToolRuleConfig(effect=WRITE, paths=("source_path", "destination_path")),
    "copy_file": ToolRuleConfig(effect=WRITE, paths=("destination_path",)),
    "read_file": ToolRuleConfig(effect=READ),
    "list_directory": ToolRuleConfig(effect=READ),
    "file_search": ToolRuleConfig(effect=READ),
}

# The text-editor tool shape (``command`` + ``path``) that Anthropic-style
# editors expose. Matched on the arguments, not on a tool name, and only for a
# tool no rule already covers. An unlisted command is an abstention: a shape we
# do not recognize must not be assumed harmless *or* mutating.
_EDITOR_WRITE_COMMANDS = frozenset({"create", "str_replace", "insert", "append", "write"})
_EDITOR_READ_COMMANDS = frozenset({"view"})


class CodingStateError(ConfigError):
    """Raised when the coding state preset cannot be built from its config."""


@dataclass(frozen=True, slots=True)
class CodingStatePlanner:
    """Select, per tool call, the sensors its own arguments make observable.

    Deterministic for the same call against the same workspace on disk: it
    resolves symlinks, so the filesystem is an input -- but it only ever reads,
    and it never runs during replay, which reads recorded evidence alone. That
    is what lets a recorded run replay identically without re-running a rule.
    """

    workspace: Path
    rules: Mapping[str, ToolRuleConfig]
    git: GitWorkspaceStateSensor | None = None
    tests: PytestStateSensor | None = None
    build: CommandStateSensor | None = None

    def plan(self, *, tool: str, args: Mapping[str, Any]) -> Sequence[StateSensor]:
        rule = self._rule(tool, args)
        if rule is None or rule.effect == READ:
            return ()
        if rule.effect == WORKSPACE:
            return self._scope()
        resources = self._resources(rule, args)
        if resources is None:
            # Ambiguous: observe what the operator configured, never a guess.
            return self._scope()
        files = tuple(
            FilesystemStateSensor(self.workspace / resource, resource_id=resource)
            for resource in resources
        )
        return files + self._scope()

    def _rule(self, tool: str, args: Mapping[str, Any]) -> ToolRuleConfig | None:
        rule = self.rules.get(tool)
        if rule is not None:
            return rule
        command = args.get("command")
        path = args.get("path")
        if not isinstance(command, str) or not isinstance(path, str):
            return None
        if command in _EDITOR_WRITE_COMMANDS:
            return ToolRuleConfig(effect=WRITE, paths=("path",))
        if command in _EDITOR_READ_COMMANDS:
            return ToolRuleConfig(effect=READ)
        return None

    def _scope(self) -> tuple[StateSensor, ...]:
        configured: list[StateSensor] = []
        if self.git is not None:
            configured.append(self.git)
        if self.tests is not None:
            configured.append(self.tests)
        if self.build is not None:
            configured.append(self.build)
        return tuple(configured)

    def _resources(self, rule: ToolRuleConfig, args: Mapping[str, Any]) -> tuple[str, ...] | None:
        """Workspace-relative ids for a rule's path arguments, or ``None`` if unusable.

        A path argument holds one path or a list of them. Every declared
        argument must resolve completely: a call that names two files and makes
        only one of them observable is attributed to neither, and a call naming
        more files than can be credibly attributed is attributed to none.
        """
        resources: set[str] = set()
        for name in rule.paths:
            value = args.get(name)
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                resource = self._resource_id(item)
                if resource is None:
                    return None
                resources.add(resource)
        if not resources or len(resources) > _MAX_OBSERVED_PATHS:
            return None
        return tuple(sorted(resources))

    def _resource_id(self, value: object) -> str | None:
        """One workspace-relative POSIX id, or ``None`` when it is not attributable.

        Absolute paths are accepted (editor-style tools use them) but only
        inside the workspace. The id is taken after symlink resolution, so two
        spellings of the same file are one resource rather than two.
        """
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_PATH_LENGTH
            or "\x00" in value
            # A backslash means different things per platform; refusing it
            # keeps one recorded resource id from depending on the host.
            or "\\" in value
        ):
            return None
        candidate = Path(value)
        target = candidate if candidate.is_absolute() else self.workspace / candidate
        try:
            resolved = resolve_non_strict(target)
        except (OSError, RuntimeError, ValueError):  # RuntimeError: symlink loop
            return None
        if self.workspace not in resolved.parents:
            return None
        try:
            return resolved.relative_to(self.workspace).as_posix()
        except ValueError:  # pragma: no cover - implied by the parents check above
            return None


def validate_preset_config(section: StateEvidenceConfig) -> None:
    """Check everything about the preset that does not depend on the machine.

    Split from :func:`build_state_producer` on purpose. The consumer side --
    ``build_pipeline``'s anchors -- reads recorded evidence only, so it must
    stay buildable where the observed workspace does not exist: replaying a
    recorded trace on another machine is a supported operation. What that side
    can still catch is every configuration mistake that is wrong anywhere: an
    unknown preset, an unusable pytest nodeid, a half-declared build command, a
    tool rule that names no argument.
    """
    if section.preset != CODING_PRESET:
        raise CodingStateError(
            f"unknown state evidence preset {section.preset!r}; the only preset is 'coding'"
        )
    _validated_timeout(section.timeout_seconds)
    _validated_rules(section.tools, section.git)
    _validated_tests(section)
    _validated_build(section)


def build_state_producer(
    section: StateEvidenceConfig, *, base_dir: Path | None = None
) -> StateEvidenceProducer:
    """Build the configured preset's producer, or raise :class:`CodingStateError`.

    The capture side, so this is where the machine-dependent checks live too: a
    workspace that does not exist, or that is not a git repository while git
    observation is on. Everything is decided before the run starts; collection
    failures at capture time stay fail-open.
    """
    validate_preset_config(section)
    workspace = _workspace(section.workspace, base_dir)
    git = _git_sensor(section, workspace)
    planner = CodingStatePlanner(
        workspace=workspace,
        rules={**BUILTIN_TOOL_RULES, **section.tools},
        git=git,
        tests=_tests_sensor(section, workspace),
        build=_build_sensor(section, workspace),
    )
    return StateEvidenceProducer(planner=planner)


def state_evidence_producer_from_config(config: PathlayerConfig) -> StateEvidenceProducer | None:
    """The producer ``detectors.state_evidence`` describes, or ``None`` if disabled.

    The capture-side companion of ``build_pipeline``'s consumer wiring, so an
    integration builds both halves of the preset from one configuration --
    mirroring :func:`~pathlayer.detectors.proof_producer.producer_from_config`.
    """
    section = config.detectors.state_evidence
    if not section.enabled:
        return None
    return build_state_producer(section)


def _workspace(configured: str, base_dir: Path | None) -> Path:
    root = (base_dir or Path.cwd()) / Path(configured)
    try:
        workspace = resolve_non_strict(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise CodingStateError(f"unusable workspace {configured!r}") from error
    if not workspace.is_dir():
        raise CodingStateError(
            f"detectors.state_evidence.workspace {configured!r} is not an existing directory"
        )
    return workspace


def _git_sensor(section: StateEvidenceConfig, workspace: Path) -> GitWorkspaceStateSensor | None:
    if not section.git:
        return None
    try:
        # Checked here so a workspace that is not a repository is a build-time
        # CodingStateError. The probe checks again, lazily, and degrades to
        # collection_failed: a repository can also break after the run starts.
        git_directory(workspace)
    except GitWorkspaceError as error:
        raise CodingStateError(str(error)) from error
    return GitWorkspaceStateSensor(GitWorkspaceProbe(workspace))


def _validated_timeout(timeout_seconds: float) -> float:
    """The budget each test or build observation gets, which must be spendable.

    The collectors require a strictly positive budget, so ``timeout_seconds: 0``
    would otherwise be accepted by the configuration and then turn every
    observation into a ``collection_failed`` at run time -- a whole run
    observing nothing, reported as a degraded collector rather than as the
    configuration mistake it is (Kropemann's review).
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CodingStateError(
            "invalid value for 'detectors.state_evidence.timeout_seconds': must be finite "
            f"and > 0, got {timeout_seconds}"
        )
    return timeout_seconds


def _validated_tests(section: StateEvidenceConfig) -> tuple[str, ...]:
    """The configured pytest selection, ordered and checked, possibly empty."""
    selection = tuple(section.tests)
    if len(selection) > MAX_OUTCOMES:
        raise CodingStateError(
            f"detectors.state_evidence.tests accepts at most {MAX_OUTCOMES} nodeids"
        )
    for nodeid in selection:
        if (
            not nodeid
            or len(nodeid) > MAX_NODEID_LENGTH
            or nodeid.startswith("-")
            or "\x00" in nodeid
        ):
            raise CodingStateError(
                f"invalid pytest nodeid {nodeid!r} in 'detectors.state_evidence.tests'"
            )
    if len(set(selection)) != len(selection):
        raise CodingStateError("detectors.state_evidence.tests must not repeat a nodeid")
    return tuple(sorted(selection))


def _validated_build(section: StateEvidenceConfig) -> tuple[str, ...]:
    """The configured validation command, checked, possibly empty."""
    if not section.build_command:
        if section.build_id:
            raise CodingStateError("detectors.state_evidence.build.id requires 'command' to be set")
        return ()
    if not section.build_id:
        raise CodingStateError("detectors.state_evidence.build.command requires an 'id'")
    try:
        return validate_command(section.build_command)
    except ValueError as error:
        raise CodingStateError(f"invalid 'detectors.state_evidence.build.command': {error}") from (
            error
        )


def _tests_sensor(section: StateEvidenceConfig, workspace: Path) -> PytestStateSensor | None:
    selection = _validated_tests(section)
    if not selection:
        return None
    return PytestStateSensor(
        test_ids=selection,
        cwd=workspace,
        resource_id=_TESTS_RESOURCE_ID,
        timeout_seconds=section.timeout_seconds,
    )


def _build_sensor(section: StateEvidenceConfig, workspace: Path) -> CommandStateSensor | None:
    command = _validated_build(section)
    if not command:
        return None
    return CommandStateSensor(
        command=command,
        cwd=workspace,
        resource_id=section.build_id,
        timeout_seconds=section.timeout_seconds,
    )


def _validated_rules(rules: Mapping[str, ToolRuleConfig], has_git: bool) -> None:
    for tool, rule in rules.items():
        where = f"detectors.state_evidence.tools.{tool}"
        if rule.effect not in TOOL_EFFECTS:
            raise CodingStateError(
                f"invalid effect {rule.effect!r} for '{where}': expected one of "
                f"{', '.join(TOOL_EFFECTS)}"
            )
        if rule.effect == WRITE and not rule.paths:
            raise CodingStateError(
                f"'{where}' declares effect 'write' but names no path argument; "
                "use effect 'workspace' when the changed paths are not in the arguments"
            )
        if rule.effect != WRITE and rule.paths:
            raise CodingStateError(f"'{where}' names path arguments but its effect is not 'write'")
        if rule.effect == WORKSPACE and not has_git:
            raise CodingStateError(
                f"'{where}' declares effect 'workspace', which requires "
                "'detectors.state_evidence.git' to be enabled"
            )
        if len(rule.paths) > _MAX_RULE_PATH_ARGUMENTS:
            raise CodingStateError(
                f"'{where}' names more than {_MAX_RULE_PATH_ARGUMENTS} path arguments"
            )


__all__ = [
    "BUILTIN_TOOL_RULES",
    "CODING_PRESET",
    "CodingStateError",
    "CodingStatePlanner",
    "build_state_producer",
    "state_evidence_producer_from_config",
    "validate_preset_config",
]
