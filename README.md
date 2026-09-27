# PlantationInference

Détection automatique des trous de plantation dans une orthophoto aérienne, et
comptage de ceux qui portent effectivement un plant.

À partir d'un GeoTIFF de parcelle, l'outil produit la position de chaque trou et
son état, puis en déduit le taux de réussite de la plantation. Un trou est classé
dans l'une des deux catégories suivantes :

- `PlantingHoleEmpty`, trou vide, dessiné en orange ;
- `PlantingHoleOccupied`, trou occupé par un plant, dessiné en vert.

## Ce que vaut le résultat

Mesuré sur 8 tuiles de `result_00389m` annotées à la main, soit 140 trous que le
modèle n'a jamais vus pendant son entraînement :

| Mesure | Valeur | Lecture |
| --- | --- | --- |
| Rappel de détection | **95,0 %** | 133 trous trouvés sur 140 |
| Précision | **85,3 %** | sur 156 boîtes tracées, 23 n'étaient pas des trous |
| Classement vide/occupé | **88,7 %** | parmi les trous trouvés |
| Bout en bout | **84,3 %** | trouvés *et* correctement étiquetés |

Le chiffre le plus utile est le dernier : environ **84 % des vrais trous sont
repérés et bien classés**.

Pour l'usage principal, estimer la réussite d'une plantation, la fiabilité est
meilleure que ces chiffres ne le laissent croire, car les erreurs de classement
se compensent : 7 trous vides annoncés occupés contre 8 occupés annoncés vides.
Sur ces tuiles, le taux réel est de 38,6 % et le modèle annonce 34,0 %, soit
**environ 5 points de marge sur l'estimation globale**.

Ce niveau convient à un diagnostic ou à un suivi de parcelle. Il ne convient pas
à un constat contractuel trou par trou sans relecture humaine.

Le détail chiffré est conservé dans le projet d'entraînement, sous
`geotiff_evaluation/result00389_run_20260922.json`.

## Installation

Il faut Python 3.10 ou plus récent, validé sur 3.12, Git, et environ 3 Go
d'espace disque pour l'environnement virtuel. Les poids du modèle sont dans le
dépôt : il n'y a aucun téléchargement séparé.

### Windows, dans PowerShell

```powershell
git clone https://github.com/HenintsoaRamananjatovo/detection-trous-plantation.git
cd detection-trous-plantation
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

### Linux ou macOS

```bash
git clone https://github.com/HenintsoaRamananjatovo/detection-trous-plantation.git
cd detection-trous-plantation
chmod +x setup.sh start_web.sh
./setup.sh
```

Les deux scripts font la même chose : créer `.venv` dans le dossier du projet,
puis y installer `requirements.txt`. Rien n'est posé ailleurs sur la machine, et
désinstaller revient à supprimer le dossier.

Deux particularités de Linux méritent d'être connues.

`pip` y livre par défaut un PyTorch compilé pour CUDA, soit plusieurs
gigaoctets de dépendances NVIDIA inutiles sans carte graphique dédiée. Pour une
machine à processeur seul :

```bash
./.venv/bin/python -m pip install torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

Et sur un serveur sans environnement graphique, `opencv-python`, installé comme
dépendance d'`ultralytics`, réclame des bibliothèques système absentes :

```bash
sudo apt install libgl1 libglib2.0-0     # Debian, Ubuntu
```

## Utilisation

Dans ce qui suit, l'interpréteur du projet s'écrit
`.\.venv\Scripts\python.exe` sous Windows et `./.venv/bin/python` ailleurs.

### Interface web

```powershell
.\start_web.ps1      # Windows
./start_web.sh       # Linux, macOS
```

Ouvrir <http://127.0.0.1:5000>, choisir un GeoTIFF, lancer. La page affiche la
progression de l'envoi puis celle du traitement, et propose les fichiers de
résultat au téléchargement.

Aucun réglage n'est exposé : les valeurs par défaut sont celles validées par la
mesure, et les rendre modifiables inviterait à les dégrader sans le savoir.

Le fichier téléversé est copié dans `uploads/`, que rien ne purge
automatiquement. Pour un raster déjà présent sur la machine, préférer la ligne
de commande, qui le lit sur place sans le dupliquer.

### Ligne de commande

```powershell
.\.venv\Scripts\python.exe cli.py "chemin\vers\parcelle.tif"   # Windows
./.venv/bin/python cli.py /chemin/vers/parcelle.tif            # Linux, macOS
```

Options utiles :

| Option | Effet |
| --- | --- |
| `--run-name` | nom du dossier de résultats, jamais écrasé |
| `--output-dir` | racine des résultats |
| `--device` | `cpu`, `0` pour le premier GPU, ou `auto` |
| `--confidence` | confiance minimale, `0.20` par défaut |
| `--no-resample` | refuser une autre échelle au lieu de la corriger |
| `--prepared-dir` | où garder les rasters rééchantillonnés |
| `--ignore-resolution` | traiter tel quel, sans corriger l'échelle |

`cli.py --help` liste les réglages fins du détecteur. Ils sont calibrés, les
modifier dégrade les résultats mesurés.

## Résultats produits

Chaque exécution crée un dossier indépendant sous `outputs/` :

| Fichier | Contenu |
| --- | --- |
| `annotated.tif` | l'orthophoto avec les boîtes dessinées |
| `detections.gpkg` | les trous en couche vectorielle, avec classe et confiance |
| `detections.csv` | une ligne par trou, coordonnées pixel et terrain |
| `summary.csv` | effectifs par classe et taux de réussite |
| `metadata.json` | source, modèle, paramètres, durée, traçabilité complète |

Dans QGIS, ajouter `annotated.tif` en couche raster et `detections.gpkg` en
couche vecteur. Le GeoPackage tient en un seul fichier, contrairement au
Shapefile.

Le taux de réussite vaut `occupés / (occupés + vides)`. C'est une estimation
issue des prédictions, pas un relevé de terrain.

## Résolution des images

Le modèle a été entraîné sur des images à **3,89 cm par pixel**, où un trou de
2 mètres occupe environ 51 pixels. C'est cette apparence qu'il reconnaît. Une
image à une autre échelle lui présente des trous d'une taille inattendue, et il
échoue.

Le pipeline s'en occupe seul. Un raster qui s'écarte de plus de 20 % de cette
valeur est rééchantillonné avant traitement, et la copie corrigée est conservée
dans `<sorties>/_prepared` pour être réutilisée ensuite.

Vérifié sur une même zone de terrain : 111 trous détectés en partant de
`result.tif` à 0,85 cm/pixel avec correction automatique, contre 113 en partant
du fichier préparé à la main, avec un nombre d'occupés identique.

**La correction ne fonctionne bien que dans un sens.** Réduire une image plus
fine ne perd rien d'utile et allège beaucoup le fichier : `result.tif` passe
ainsi de 1720 à 262 Mo. Agrandir une image plus grossière rétablit la taille
apparente mais pas le détail, qui n'a jamais été enregistré. Au-delà de
**10 cm par pixel** le raster est donc refusé, avec un message qui l'explique.

Les mesures de dégradation en fonction de la résolution sont conservées dans le
projet d'entraînement, sous `geotiff_evaluation/limites_resolution.json`.

## Modèle

Les poids utilisés sont dans `models/`, copiés depuis le projet d'entraînement
pour que celui-ci soit déployable seul.

Un seul réseau fait tout : `direct_two_classes_best.pt`, un YOLO11s de détection
qui produit en une passe la boîte et sa classe. Il n'y a pas d'étape de
classification séparée.

`occupancy_classifier_best.pt` est le vestige d'une architecture en cascade
abandonnée, où un second réseau reclassait chaque vignette découpée. Mesurée sur
les deux rasters annotés, elle dégradait le classement, de 0,935 à 0,908 sur
`source2` et de 0,887 à 0,827 sur `result00389`, pour un temps doublé. Ces poids
ne servent plus qu'à refaire la comparaison avec `--classifier`.

## Fonctionnement interne

Le raster est lu par fenêtres successives, sans jamais écrire de tuiles
intermédiaires sur le disque. Chaque fenêtre passe au détecteur, puis les boîtes
sont ramenées dans le repère du raster complet.

Trois corrections sont appliquées, toutes issues de mesures :

- les boîtes sont réduites d'un facteur `0.82`, car la rotation libre utilisée
  pendant l'entraînement les fait systématiquement surestimer ;
- les boîtes plus de deux fois plus longues que larges sont écartées, ce sont
  presque toujours des faux positifs sur le sol nu entre les rangées ;
- une suppression des doublons, indépendante de la classe, s'applique sur tout
  le raster, ce qui évite de compter deux fois un trou à cheval sur deux
  fenêtres. Sur une parcelle réelle, cela retire 347 doublons sur 2864 boîtes.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q   # Windows
./.venv/bin/python -m pytest -q           # Linux, macOS
```

29 tests couvrent le découpage, la déduplication, les filtres de forme, le
rééchantillonnage automatique et son cache, le téléversement et ses refus. Ils
fabriquent leurs propres rasters et n'ont besoin d'aucune donnée externe.

## Contenu du projet

| Élément | Rôle |
| --- | --- |
| `app.py` | interface web |
| `cli.py` | ligne de commande et valeurs par défaut |
| `plantation_inference/` | le pipeline : découpage, détection, préparation, sorties |
| `templates/` | la page web |
| `models/` | les poids du réseau |
| `outputs/` | les résultats, un dossier par exécution |
| `tests/` | la suite de tests |
| `setup.ps1`, `setup.sh` | installation, Windows et Unix |
| `start_web.ps1`, `start_web.sh` | lancement de l'interface |
