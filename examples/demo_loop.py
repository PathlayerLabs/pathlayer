"""DEMO-1 -- une boucle detectee et stoppee par le vrai chemin ACT-0.

La plus petite histoire complete de Pathlayer : un agent se bloque et refait
sans cesse les memes gestes. Sans garde-fou il tournerait sans fin ; avec
Pathlayer la boucle est detectee et l'agent est stoppe proprement.

Le stop passe par le chemin runtime reel, pas un raccourci :

    agent -> wire(pipeline, actuator) -> LoopDetector -> Policy -> STOP
          -> DemoActuator.apply(STOP) -> l'agent draine et s'arrete

``DemoActuator`` est le pendant offline de ``LangGraphActuator`` : sur STOP il
demande un drain coopératif (un flag) que l'agent observe. Il ne decide jamais
-- l'``Action`` lui arrive deja arbitree par le pipeline.

Une contrepartie saine (outils varies, bornee) est cablee de la meme facon et
n'est PAS stoppee : le cout d'un faux positif est central.

Deterministe, offline, sans LLM ni reseau. Lancer :

    python examples/demo_loop.py
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from pathlayer import Pipeline
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.observability import JsonFormatter, get_logger
from pathlayer.policies import Action, Actuator, wire
from pathlayer.postmortem import PostMortem

# Sans garde-fou l'agent bloque ne s'arreterait jamais. On borne la demo pour
# qu'elle finisse, tout en montrant qu'a ce cap il "tourne toujours".
NAIVE_CAP = 200


@dataclass
class DemoRunControl:
    """Drapeau de drain coopératif : le pendant offline de LangGraph RunControl.

    L'actuator demande le drain (``stopped``) ; l'agent l'observe apres chaque
    pas et s'arrete de lui-meme. Aucun framework, aucun reseau.
    """

    stopped: bool = False
    reason: str = ""


class DemoActuator(Actuator):
    """Execute STOP en demandant un drain sur un ``DemoRunControl``.

    Calque de ``LangGraphActuator`` : n'agit que sur ``Action.STOP`` et ne
    decide jamais -- l'``Action`` arrive deja arbitree par le pipeline.
    """

    def __init__(self, control: DemoRunControl) -> None:
        self._control = control

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is not Action.STOP:
            return
        self._control.stopped = True
        self._control.reason = post_mortem.reason if post_mortem is not None else "stop"


def looping_agent() -> Iterator[ToolEvent]:
    """Agent bloque : il liste puis relit le meme fichier, sans jamais avancer.

    Chaque pas porte un ``ts`` distinct (evenements non byte-identiques) ; la
    canonicalisation les rapproche pour que la boucle soit visible.
    """
    step = 0
    while True:
        for tool in ("list_dir", "read_file"):
            yield ToolEvent(tool=tool, args={"path": "src/main.py"}, ts=float(step))
            step += 1


def healthy_agent() -> Iterator[ToolEvent]:
    """Agent sain et borne : des outils varies qui font avancer la tache, puis fin."""
    plan: list[tuple[str, dict[str, object]]] = [
        ("list_dir", {"path": "src"}),
        ("read_file", {"path": "src/main.py"}),
        ("edit_file", {"path": "src/main.py", "patch": "add guard"}),
        ("run_tests", {"target": "tests/test_main.py"}),
        ("edit_file", {"path": "src/main.py", "patch": "fix off-by-one"}),
        ("run_tests", {"target": "tests/test_main.py"}),
        ("git_commit", {"message": "fix guard and off-by-one"}),
    ]
    for i, (tool, args) in enumerate(plan):
        yield ToolEvent(tool=tool, args=args, ts=float(i))


def _run_naive(agent: Iterator[ToolEvent], cap: int) -> int:
    """Sans Pathlayer : l'agent avance jusqu'au cap (ou jusqu'a s'arreter seul)."""
    executed = 0
    for _ in agent:
        executed += 1
        if executed >= cap:
            break
    return executed


def _run_guarded(
    agent: Iterator[ToolEvent], logger: logging.Logger
) -> tuple[Pipeline, DemoRunControl, int]:
    """Avec Pathlayer (ACT-0) : chaque pas traverse ``wire()`` ; STOP draine l'agent."""
    pipeline = Pipeline([LoopDetector()], logger=logger)
    control = DemoRunControl()
    sink = wire(pipeline, DemoActuator(control), logger=logger)
    executed = 0
    for event in agent:
        sink(event)
        executed += 1
        if control.stopped:
            break
        if executed >= NAIVE_CAP:  # filet de securite si la detection echouait
            break
    return pipeline, control, executed


def _triggering_detector(pm: PostMortem) -> str:
    if not pm.signals:
        return "?"
    return max(pm.signals, key=lambda s: s.confidence).name


def scenario_loop(logger: logging.Logger) -> None:
    print("--- BOUCLE : agent bloque ---")
    naive = _run_naive(looping_agent(), NAIVE_CAP)
    pipeline, control, _ = _run_guarded(looping_agent(), logger)
    pm = pipeline.post_mortem

    print(f"Sans Pathlayer : {naive} actions et il tourne toujours (cap demo {NAIVE_CAP}).")
    if pm is None:
        print("Avec Pathlayer : rien detecte (INATTENDU pour cette trace).\n")
        return
    print(f"Avec Pathlayer : STOP apres {pm.events_processed} actions.")
    print(f"   Detecteur declencheur : {_triggering_detector(pm)} ({pm.reason})")
    print(f"   Pas de detection      : action #{pm.stopped_at_index}")
    print(f"   Actuator             : {'drain demande' if control.stopped else 'non declenche'}")
    avoided = naive - pm.events_processed
    print(f"   Travail evite        : {avoided} actions sur {naive}.\n")


def scenario_healthy(logger: logging.Logger) -> None:
    print("--- SAIN : agent qui avance (contre-exemple) ---")
    pipeline, control, executed = _run_guarded(healthy_agent(), logger)

    if pipeline.stopped or control.stopped:
        pm = pipeline.post_mortem
        reason = pm.reason if pm is not None else "?"
        print(f"Avec Pathlayer : STOP apres {executed} actions (FAUX POSITIF : {reason}).\n")
        return
    print(f"Avec Pathlayer : {executed} actions, aucune intervention. L'agent finit seul.\n")


def main() -> None:
    # OBS-1 : signal / decision / action en JSON lines vers stderr, pour voir
    # exactement quand et pourquoi Pathlayer agit sans polluer la sortie demo.
    logger = get_logger("pathlayer.demo")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    print("=" * 60)
    print("DEMO-1 PATHLAYER - une boucle detectee et stoppee (ACT-0)")
    print("=" * 60 + "\n")
    scenario_loop(logger)
    scenario_healthy(logger)


if __name__ == "__main__":
    main()
