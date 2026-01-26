# Projet PLE M2

Repository du projet de l'UE PLE (Programmation Large Échelle) de Master 2.

## Description

Projet d'analyse de données Clash Royale utilisant Hadoop MapReduce pour:
- **Data Cleaning**: Nettoyage et validation des données brutes
- **Nodes/Edges Generation**: Génération de graphes d'archétypes et matchups
- **Statistics**: Analyse statistique des résultats

## Usage

### Data Cleaning

Nettoyage des données avec nombre de reducers configurable (2):

```sh
yarn jar projet_ple-1.jar clean /user/auber/data_ple/clash_royale/raw_data_100K.json output 2
```

### Nodes/Edges Generation

Génération des archétypes (nodes) et matchups (edges) avec **configuration flexible des reducers**:

**Mode 1 - Valeurs par défaut (3 arguments):**
```sh
yarn jar projet_ple-1.jar nodesedges input output 7
# NODES: 1 reducer, EDGES: 10 reducers
```

**Mode 2 - Contrôle EDGES seulement (4 arguments):**
```sh
yarn jar projet_ple-1.jar nodesedges full full_7 7 150
# NODES: 1 reducer, EDGES: 150 reducers
```

**Mode 3 - Contrôle complet (5 arguments):**
```sh
yarn jar projet_ple-1.jar nodesedges full full_7 7 2 150
# NODES: 2 reducers, EDGES: 150 reducers
```

### Statistics

Analyse statistique des résultats:

```sh
yarn jar projet_ple-1.jar stats input output
```

Récupération des fichiers en un seul sur la gateway temporairement:
```sh
hdfs dfs -getmerge input output
```


# Authors

BONNEFOUS Guilhem
LAFFARGUE Alexandre
