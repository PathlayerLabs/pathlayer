"""Demo avant/apres : trois facons pour un agent de deraper, et Pathlayer qui coupe.

Une seule commande montre les trois detecteurs reels du Sprint 1 en action :
- BOUCLE  : l'agent refait sans cesse les memes gestes (`LoopDetector`).
- RETRY   : l'agent rejoue le meme appel qui echoue (`RetryDetector`).
- BUDGET  : l'agent brule le budget en dollars (`BudgetDetector`).

Dans chaque cas, sans garde-fou l'agent continuerait ; avec Pathlayer il est
stoppe proprement, avec une explication (post-mortem).

Lancer :

    python examples/demo_loop.py
"""

from __future__ import annotations

from pathlayer import Pipeline
from pathlayer.detectors.base import Detector
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent

NAIVE_CAP = 50


def _run_with_pathlayer(events: list[ToolEvent], detectors: list[Detector]) -> Pipeline:
    pipeline = Pipeline(detectors)
    pipeline.run(events)
    return pipeline


def _report(pipeline: Pipeline, total: int, naive: str) -> None:
    pm = pipeline.post_mortem
    print(f"Sans Pathlayer (attendu) : {naive}")
    if pm is None:
        print("Avec Pathlayer : rien detecte (inattendu pour cette trace).\n")
        return
    print(f"Avec Pathlayer : stop apres {pm.events_processed} actions sur {total}.")
    print(f"   Raison       : {pm.reason}")
    print(f"   Economie     : {total - pm.events_processed} actions evitees sur cette trace.\n")


def scenario_loop() -> None:
    print("--- BOUCLE (LoopDetector) ---")
    # instances fraiches a chaque repetition (pas de references partagees)
    events = [
        ToolEvent(tool=tool, args={"path": "a.py"}) for _ in range(30) for tool in ("read", "edit")
    ]
    executed = min(len(events), NAIVE_CAP)
    naive = f"{executed}+ actions, il tourne en rond sans jamais s'arreter."
    _report(_run_with_pathlayer(events, [LoopDetector()]), len(events), naive)


def scenario_retry() -> None:
    print("--- RETRY (RetryDetector) ---")
    # instances fraiches (pas de reference partagee entre les 6 echecs)
    events = [
        ToolEvent(tool="run_tests", args={}, result={"error": "AssertionError"}) for _ in range(6)
    ]
    naive = f"{len(events)} fois le meme appel qui echoue, en esperant un autre resultat."
    _report(_run_with_pathlayer(events, [RetryDetector()]), len(events), naive)


def scenario_budget() -> None:
    print("--- BUDGET (BudgetDetector) ---")
    events = [ToolEvent(tool="call_llm", args={}, cost_usd=1.5) for _ in range(6)]
    naive = f"{len(events)} appels a $1.50, le budget de $5 explose sans alerte."
    _report(_run_with_pathlayer(events, [BudgetDetector(max_cost_usd=5.0)]), len(events), naive)


def main() -> None:
    print("=" * 60)
    print("DEMO PATHLAYER - trois derapages, trois freins")
    print("=" * 60 + "\n")
    scenario_loop()
    scenario_retry()
    scenario_budget()


if __name__ == "__main__":
    main()
