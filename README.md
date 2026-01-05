# Projet PLE M2

Repository du projet de l'UE PLE (Programmation Large Échelle) de Master 2.

# Usage

Utiliser la commande :

Data cleaning avec 2 reducers :

```sh
yarn jar projet_ple-1.jar clean /user/auber/data_ple/clash_royale/raw_data_100K.json test 2
```

Nodesedges pour achétypes de taille 7 avec 40 reducers :

```sh
yarn jar projet_ple-1.jar nodesedges full full_7 7 40
```

Stats pour le 1M de taille 7 :

```sh
yarn jar projet_ple-1.jar stats 1M_7 1M_stats_7
```

# Authors

BONNEFOUS Guilhem
LAFFARGUE Alexandre
