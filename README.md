# Macro Engine Dashboard

Dashboard web statique du signal mensuel SPY Risk-On / Risk-Off.

## Fichiers principaux
- `generate_dashboard.py` : calcule le signal et régénère `docs/index.html` + `docs/signal_snapshot.json`
- `.github/workflows/update_dashboard.yml` : lance le calcul automatiquement en ligne via GitHub Actions
- `requirements.txt` : dépendances Python
- `docs/` : fichiers publiés par GitHub Pages

## Publication 100% en ligne
1. Créer un repo GitHub vide
2. Ajouter tous les fichiers
3. Activer GitHub Pages sur la branche `main`, dossier `/docs`
4. Activer GitHub Actions
5. Lancer une première exécution via `Actions > Update Macro Engine Dashboard > Run workflow`

Le workflow tourne ensuite chaque jour mais ne met à jour les fichiers que le dernier jour du mois.
