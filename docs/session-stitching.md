# Répétitions au niveau d'une session

Activez `pipeline.session_stitching: true` dans la configuration Pathlayer pour reconnaître des actions équivalentes entre plusieurs exécutions liées. La fonction est désactivée par défaut.

Les seuils et signaux existants sont conservés. Seul le détecteur de retry utilise ce contexte; boucles, budgets, watchdogs, état de politique, absence de progrès, cycle de vie et STOP restent propres à chaque exécution.

## Capture

OpenClaw utilise uniquement le `ctx.sessionKey` fourni par la source. LangGraph exige un identifiant de session et un identifiant d'exécution logique fournis explicitement par l'appelant. Une identité absente ou ambiguë ne produit aucune conclusion inter-exécutions. Pathlayer ne détecte pas qu'une même identité a été réutilisée pour une autre tâche : il revient à la source ou à l'appelant de fournir une identité propre à la tâche. Un échec identique répété sous une identité réutilisée serait donc joint.

Chaque invocation conserve donc son pipeline et son cycle de vie indépendants. La fin d'une exécution ne ferme pas la session source.

Seuls des échecs identiques sont joints : même action canonique et même erreur. Un appel réussi n'est jamais joint, car un même appel réussi répété d'un tour à l'autre vient le plus souvent de l'utilisateur qui relance (attente d'une CI, question reposée), pas d'un agent bloqué. Une erreur différente ou un succès rompt donc la continuité.

L'erreur ne suffit pas à elle seule : une observation d'état inchangé et valide, ou une preuve complète de checkpoints encore non satisfaits, est aussi nécessaire pour joindre deux actions. Un état modifié, une preuve partielle ou satisfaite, une preuve invalide ou indisponible, une action différente, une liaison manquante ou un résultat de test rompt la continuité. La sortie de l'outil seule ne constitue jamais une preuve.

## Durabilité et replay

Le journal append-only se trouve dans `<store.path>.sessions`. Chaque capture est isolée par source, session, exécution logique et identifiant d'événement. Les captures identiques sont idempotentes; un payload différent avec la même identité est rejeté.

Le contexte borné utilisé pour la décision est attaché à l'événement capturé. Le replay reproduit donc la décision sans relire le journal courant ni sonder à nouveau l'espace de travail. Les payloads bruts et les décisions canoniques ne sont pas réécrits. Les empreintes de configuration historiques restent compatibles; l'activation de cette fonction possède une version de sémantique distincte.

Les traces étiquetées de `traces/val7/` portent ce contexte figé, qui ne compte qu'avec l'option activée. Elles sont rejouées avec `session_stitching` par `tests/test_session_stitching.py` et exclues du score agrégé de `python -m pathlayer.validation.replay traces`, qui utilise la configuration par défaut.

Une observation manquée insère une rupture de fenêtre dans le journal : aucune répétition n'est jointe par-dessus, et la continuité reprend avec les observations capturées ensuite. Trois cas produisent une rupture : une erreur de capture, une exécution OpenClaw interrompue (son suffixe non observé est inconnu) et un changement de configuration (les observations ne sont plus comparables, y compris lors d'un retour à l'ancienne configuration). La session n'est jamais désactivée définitivement et il n'est pas nécessaire de changer d'identité de session.

Si le journal est inaccessible, la rupture est conservée dans un marqueur synchronisé sur disque, un fichier par observation manquée, puis inscrite dans le journal avant la capture suivante de cette session. Le traitement local continue normalement. Si même le marqueur ne peut pas être écrit, l'erreur est propagée plutôt que de prétendre à une continuité qui survivrait à un redémarrage. Une série d'échecs de capture ne produit qu'un avertissement, jusqu'à la prochaine capture réussie.

## Changements actifs même sans l'option

Trois changements de capture livrés avec cette évolution s'appliquent même quand `session_stitching` est désactivé :

- `LangGraphAdapter.from_config` branche le producteur de preuve ANC-4 (`producer_from_config`) dès que `detectors.progress_proof` est activé : la sonde s'exécute alors après chaque appel d'outil. Pour s'en passer, construisez l'adaptateur directement sans `progress_proof`.
- Si ce producteur lève une exception côté LangGraph, l'événement porte `progress_proof_unobserved: {"reason": "producer_failed"}`. Ce marqueur sort volontairement du schéma fermé ANC-4 : `ProofProgressAnchor` l'ignore et se comporte comme sans marqueur, mais sa présence suffit à empêcher toute jonction inter-exécutions.
- Le manifeste OpenClaw conserve la `session_key` fixée à la création du run ; le hook de démarrage ne la réécrit plus avec son propre `ctx.sessionKey`.

## Rapport en lecture seule

```bash
python -m pathlayer.session --source openclaw --session-key ma-session \
  runs/tour-un/trace.json runs/tour-deux/trace.json runs/tour-trois/trace.json
```

Ajoutez `--json` pour obtenir une sortie structurée et `--config` pour utiliser des masques de canonicalisation personnalisés. Le même regroupement est disponible avec `SessionJournal.report()` et `project_session(...)`.

Le rapport décrit un échec répété comparable observé: identité canonique, outil borné, nombre d'occurrences, identifiants de runs et index d'observation. Il renvoie vers les événements et décisions propres à chaque run, sans créer de nouvelle alerte ni inventer l'attribution d'un détecteur.

## Limites

Cette version joint uniquement des échecs consécutifs et équivalents, selon la canonicalisation existante. Une répétition réussie entre plusieurs tours n'est jamais signalée au niveau de la session, même si elle traduit une vraie boucle; la détection propre à chaque exécution reste inchangée. Elle ne joint pas encore les cycles alternant plusieurs outils et ne transporte pas l'état général d'absence de progrès entre les runs. Une preuve inconnue abstient le signal inter-session; la détection propre à l'exécution continue normalement.

Le contexte est limité à 64 observations et 128 KiB d'événements encodés. Une limite dépassée crée une rupture de fenêtre. Ce sont des bornes de stockage et de replay, pas des seuils calibrés sur une nouvelle métrique.

Le journal contient des données de traces sensibles et doit recevoir les mêmes protections d'accès que les traces existantes. La rétention et la compaction restent hors du périmètre de cette évolution.
