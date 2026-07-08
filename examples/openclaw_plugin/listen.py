"""Ecouteur pret a l'emploi pour valider l'adapter OpenClaw (ADP-1) en vrai.

Ce script ne demande AUCUNE ligne de Python a ecrire ou modifier : il demarre
`OpenClawAdapter`, affiche le port sur lequel il ecoute, puis imprime chaque
evenement recu en direct, au fur et a mesure qu'OpenClaw les envoie via le
plugin de reference (`pathlayer-plugin/`, dans ce meme dossier).

A lancer sur la MEME machine que le gateway OpenClaw (voir README.md de ce
dossier pour le runbook complet) :

    python examples/openclaw_plugin/listen.py

Pour qu'il survive a une deconnexion SSH, le lancer avec nohup :

    nohup python examples/openclaw_plugin/listen.py > pathlayer.log 2>&1 &
    tail -f pathlayer.log

Arret : Ctrl+C si lance au premier plan, ou `pkill -f listen.py` si lance avec
nohup en arriere-plan.
"""

from __future__ import annotations

import os
import signal
import sys
import types
from datetime import datetime, timezone

from pathlayer.adapters.openclaw import DEFAULT_PORT, OpenClawAdapter
from pathlayer.events import ToolEvent

# Permet de changer de port sans toucher au code, si 8787 est deja pris sur la
# VM -- dans ce cas, penser a mettre a jour PATHLAYER_URL du cote du plugin
# JavaScript (voir pathlayer-plugin/index.js) avec le meme numero.
PORT = int(os.environ.get("PATHLAYER_PORT", DEFAULT_PORT))


def _print_event(event: ToolEvent) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] outil = {event.tool!r}")
    print(f"           args   = {event.args}")
    if event.result is not None:
        print(f"           result = {event.result!r}")
    if event.metadata.get("error") is not None:
        print(f"           ERREUR = {event.metadata['error']!r}")
    print(flush=True)


def main() -> None:
    adapter = OpenClawAdapter(_print_event, port=PORT)
    adapter.attach()

    print("=" * 60)
    print("Ecouteur Pathlayer pour OpenClaw (ADP-1) -- pret.")
    print("=" * 60)
    print(f"En ecoute sur : http://127.0.0.1:{adapter.port}/ingest")
    print("Le plugin OpenClaw doit pointer vers cette meme URL (voir")
    print("pathlayer-plugin/index.js : PATHLAYER_URL / PATHLAYER_PORT).")
    print()
    print("Demande maintenant une tache simple a OpenClaw (ex: lire un")
    print("fichier) et regarde les evenements apparaitre ci-dessous.")
    print("Ctrl+C pour arreter.\n", flush=True)

    def _stop(_signum: int, _frame: types.FrameType | None) -> None:
        print("\nArret de l'ecouteur.")
        adapter.detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.pause()


if __name__ == "__main__":
    main()
