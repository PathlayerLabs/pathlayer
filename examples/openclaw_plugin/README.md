# Runbook Xavier -- tester l'adapter OpenClaw (ADP-1)

Ce guide ne demande aucune lecture de code : copie-colle les commandes dans
l'ordre. Tout se passe **sur la VM Lightsail qui fait tourner OpenClaw**, pas
sur ton Mac -- ouvre une deuxieme session SSH navigateur Lightsail (en plus de
celle qui fait deja tourner OpenClaw) pour Pathlayer.

Si une etape ne se passe pas comme prevu, arrete-toi et renvoie-nous
exactement le texte affiche dans le terminal -- pas besoin de comprendre ce
que ca veut dire.

## Etape 0 -- deux infos avant de commencer

Dans le terminal SSH d'OpenClaw :

```bash
openclaw --version
free -h
```

Renvoie-nous ces deux resultats avant de continuer (ca evite un aller-retour
si l'installation du plugin coince a l'etape 3).

## Etape 1 -- recuperer le code Pathlayer sur la VM

Dans la deuxieme session SSH (celle dediee a Pathlayer) :

```bash
# Verifie que l'acces au depot fonctionne AVANT d'aller plus loin -- si cette
# ligne demande un mot de passe ou reste bloquee, arrete-toi et dis-le nous
# plutot que d'attendre :
git ls-remote git@github.com:Kropemann/Pathlayer.git >/dev/null && echo "acces OK" || echo "ACCES A CONFIGURER D'ABORD"

# Si pathlayer n'est pas encore present sur la VM :
git clone git@github.com:Kropemann/Pathlayer.git pathlayer
cd pathlayer
# Si le dossier existe deja (tu l'avais peut-etre deja clone la semaine
# derniere) :
cd pathlayer && git fetch origin

git checkout feat/40-openclaw-http-adapter
git pull

# Environnement Python isole (evite une erreur "externally-managed-environment"
# frequente sur les images Ubuntu recentes) :
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Etape 2 -- lancer l'ecouteur Pathlayer

Toujours dans cette meme session SSH (celle de Pathlayer) :

```bash
nohup python examples/openclaw_plugin/listen.py > pathlayer.log 2>&1 &
tail -f pathlayer.log
```

Tu dois voir un message du genre `En ecoute sur : http://127.0.0.1:8787/ingest`.

Laisse tourner. `Ctrl+C` n'arrete que l'affichage (`tail`), pas l'ecouteur --
il continue meme si la session SSH se coupe. Pour l'arreter pour de vrai plus
tard : `pkill -f listen.py`.

## Etape 3 -- installer le plugin dans OpenClaw

Dans le terminal SSH d'OpenClaw cette fois, copie le dossier du plugin (celui
qui est arrive avec le code a l'etape 1) dans le dossier des extensions
d'OpenClaw, puis redemarre le gateway :

```bash
cp -r ~/pathlayer/examples/openclaw_plugin/pathlayer-plugin ~/.openclaw/extensions/pathlayer-plugin
openclaw gateway restart
openclaw gateway status
```

(Adapte `~/pathlayer` si tu as clone ailleurs, et `~/.openclaw/extensions` si
ton install utilise un autre chemin -- `openclaw --version` a l'etape 0 nous
dira si le chemin est different sur ta version.)

## Etape 4 -- comment revenir en arriere si besoin

Avant de tester quoi que ce soit, garde cette commande sous la main : elle
retire le plugin et redemarre le gateway proprement, sans toucher a ta
configuration OpenClaw existante (celle que tu as mis du temps a onboarder) :

```bash
rm -rf ~/.openclaw/extensions/pathlayer-plugin && openclaw gateway restart
```

## Etape 5 -- tester

1. Depuis ton Mac, ouvre le dashboard OpenClaw comme d'habitude (via le tunnel
   VPN).
2. Demande-lui une tache **anodine** (par exemple "lis le fichier README.md
   et dis-moi combien de lignes il fait") -- evite un fichier sensible,
   puisque tu vas nous coller la sortie telle quelle.
3. Retourne voir `pathlayer.log` sur la VM (`tail -f pathlayer.log` si tu ne
   l'as plus sous les yeux) : un evenement doit apparaitre pour chaque outil
   utilise par OpenClaw.
4. Renvoie-nous :
   - le contenu de `pathlayer.log` (meme si vide -- ca veut dire quelque
     chose aussi) ;
   - les logs d'OpenClaw autour du moment du test (le plugin y ecrit une
     ligne `[pathlayer] ...` a chaque tentative d'envoi, succes ou echec --
     ca nous dit ou ca coince si ca ne marche pas).

Si `pathlayer.log` reste vide **et** qu'aucune ligne `[pathlayer] plugin
registered...` n'apparait dans les logs d'OpenClaw juste apres le redemarrage
du gateway (etape 3), c'est probablement que la methode d'enregistrement du
hook (`api.registerHook` dans `index.js`) ne correspond pas a l'API de ta
version d'OpenClaw -- dis-le nous, c'est un point precis a ajuster cote
plugin, pas la peine de chercher plus loin de ton cote.

## Notes

- Le plugin (`pathlayer-plugin/index.js`) est ecrit contre la documentation
  publique d'OpenClaw, jamais teste contre un vrai gateway depuis ce depot --
  attends-toi a devoir l'ajuster une fois ton retour recu, ce n'est pas
  grave.
- Rien ici ne bloque ou ne ralentit l'agent : si Pathlayer est eteint ou lent,
  le plugin abandonne l'envoi au bout de 300 ms et OpenClaw continue
  normalement.
