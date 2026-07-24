# DEMO-2 — l'agent qui tourne sans avancer (le cas « 47 000 $ »)

Ce document explique, sans jargon, ce que montre la démo `demo_no_progress.py`
et comment la lancer. Il est fait pour être partagé tel quel, y compris à
quelqu'un qui ne code pas : tu peux te contenter de lire la section
« Ce que ça prouve » et de regarder la sortie plus bas, sans rien installer.

---

## En une phrase

Un agent IA peut avoir l'air occupé — il fait un geste différent à chaque tour —
tout en n'avançant jamais vraiment. Il ne se répète pas, donc les garde-fous
classiques ne voient rien. Pathlayer, lui, s'en rend compte et intervient de
façon graduée : il **prévient**, puis **force un bilan**, puis **arrête** l'agent.

C'est le scénario réel qui a coûté environ 47 000 $ à une équipe : deux agents
qui se sont parlé pendant 11 jours sans jamais crasher ni tourner en rond — ils
ne se sont simplement jamais arrêtés.

---

## Ce que ça prouve

- **Pathlayer attrape un problème que la détection de boucle ne voit pas.**
  L'agent change d'action à chaque tour ; seule la mesure du *vrai* progrès (les
  tests qui passent) révèle qu'il est bloqué.
- **L'intervention est douce, pas brutale.** On ne coupe pas tout de suite : on
  avertit, on force un bilan de progrès, et seulement si rien ne bouge on arrête.
- **Pas de faux positif.** Un agent qui progresse vraiment (même type d'activité,
  mais les tests passent de plus en plus) n'est **jamais** dérangé : il finit
  seul. Couper un agent qui travaille bien serait pire que de ne rien avoir.

Le tout tourne **en local, sans IA dans la boucle, sans réseau** : c'est
déterministe et reproductible.

---

## La sortie que tu obtiens

```text
================================================================
DEMO-2 PATHLAYER - no-progress slow burn detecte et stoppe (ACT-1/2/0)
================================================================

--- SLOW BURN : busy agent, zero progress (the differentiated case) ---
Sans Pathlayer : 200+ actions et il tourne toujours (cap demo 200).
Avec Pathlayer : STOP (drain execute).
   Repetition silencieuse : loop=0.00  retry=0.00 (attendu ~0)
   No-progress detecte    : no_progress=1.00 (monte et sature)
   Premiere intervention  : action #5
   Ladder gradue          : WARN -> SUMMARIZE -> STOP
   Resume force execute   : oui (pathlayer_progress_summary)
   Raison du STOP         : no_progress: 11 steps without new progress (cusum)
   Travail evite          : ~186 actions sur 200 (14 traitees).

--- SAIN : meme activite, mais les tests avancent (contre-exemple) ---
Avec Pathlayer : 8 actions, no_progress=0.10, aucune intervention. L'agent finit seul.
```

Comment lire ça :

- **`loop=0.00 retry=0.00`** : les détecteurs de répétition restent muets — c'est
  normal, l'agent ne se répète pas. C'est tout l'intérêt du cas.
- **`no_progress=1.00`** : le détecteur de progrès, lui, est certain que rien
  n'avance.
- **`WARN -> SUMMARIZE -> STOP`** : les trois paliers d'intervention, dans l'ordre.
- **`~186 actions sur 200` évitées** : ce que Pathlayer a économisé.
- Le bloc **SAIN** montre le contre-exemple : aucune intervention.

---

## Comment la lancer soi-même

Il faut Python 3.10+ installé.

```bash
# 1. Récupérer le projet
git clone https://github.com/Kropemann/Pathlayer.git
cd Pathlayer

# 2. Créer un environnement isolé et installer Pathlayer (+ le support LangGraph)
python -m venv .venv
source .venv/bin/activate          # sous Windows : .venv\Scripts\activate
pip install -e ".[langgraph]"

# 3. Lancer la démo
python examples/demo_no_progress.py
```

La démo se termine en une dizaine de secondes et affiche exactement la sortie
ci-dessus.

---

## Ce que la démo ne prétend PAS faire

Elle démontre le cas d'un agent qui produit un **signal de progrès mesurable**
(ici : le résultat des tests). Elle ne prétend pas attraper un agent bloqué qui
n'a *aucun* repère objectif (ni tests, ni suivi de code, ni code de sortie). Ce
cas plus difficile est suivi séparément (issue ANC-1, #135).

---

## Pour aller plus loin (technique)

Le détail de l'architecture (adapter LangGraph, signaux `advisory`, CUSUM, chemin
runtime complet) est documenté dans [`examples/README.md`](../examples/README.md),
section `demo_no_progress.py (DEMO-2)`.
