# analyse_stats_light_optimized.py
import pandas as pd
import numpy as np
from scipy import stats
import glob
import sys
import os
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Analyse des statistiques de matchs d'archétypes")
    parser.add_argument("stats_dir", nargs="?", default="./stats_local", help="Chemin répertoire stats")
    parser.add_argument("--output-sample", default="sample_for_viz.csv", help="Fichier sortie échantillon")
    parser.add_argument("--output-json", default="stats_results.json", help="Fichier sortie JSON")
    
    # J'ai ajouté un argument pour contrôler le taux de lecture
    parser.add_argument("--read-fraction", type=float, default=0.1, 
                        help="Fraction des données à lire (ex: 0.1 pour 10%). 1.0 pour tout lire.")
    
    parser.add_argument("--sample-size", type=int, default=100000, help="Taille de l'échantillon final sauvegardé")

    args = parser.parse_args()

    stats_dir = args.stats_dir
    output_sample = args.output_sample
    output_json = args.output_json
    read_fraction = args.read_fraction # Nouveau paramètre
    max_sample_size = args.sample_size

    print(f"📂 Lecture des fichiers depuis {stats_dir}...")
    
    if not os.path.exists(stats_dir):
        print(f"❌ Répertoire introuvable: {stats_dir}")
        sys.exit(1)

    files = glob.glob(os.path.join(stats_dir, "part-*"))
    if not files:
        print(f"❌ Aucun fichier part-* trouvé")
        sys.exit(1)

    print(f"📂 Lecture de {len(files)} fichier(s)...")
    if read_fraction < 1.0:
        print(f"⚠️  Mode ÉCONOMIE DE RAM : Lecture de {read_fraction*100}% des données aléatoirement.")

    # Optimisation des types pour réduire la RAM par 2
    dtypes = {
        "source": "string", # Ou 'category' si peu de valeurs uniques, mais 'string' est plus sûr ici
        "target": "string",
        "count": "int32",       # int32 suffit pour des milliards (jusqu'à 2e9)
        "wins": "int32",
        "count_source": "int32",
        "count_target": "int32",
        "prediction": "float32" # float32 suffit largement pour des probas
    }

    chunks = []
    total_rows_seen = 0
    
    for f in files:
        try:
            # On lit par morceaux
            for chunk in pd.read_csv(
                f,
                sep=";",
                names=["source", "target", "count", "wins", "count_source", "count_target", "prediction"],
                dtype=dtypes, # Application des types optimisés
                chunksize=100000,
                on_bad_lines='skip' # Évite de planter sur une ligne corrompue
            ):
                # ÉCHANTILLONNAGE À LA VOLÉE
                # Si on a trop de données, on ne garde qu'une fraction aléatoire DU CHUNK
                if read_fraction < 1.0:
                    chunk = chunk.sample(frac=read_fraction, random_state=42)
                
                chunks.append(chunk)
                total_rows_seen += len(chunk)
                
                # Sécurité : Si on dépasse 50 millions de lignes en mémoire, on prévient
                if total_rows_seen > 50_000_000 and len(chunks) % 100 == 0:
                    print(f"   ... {total_rows_seen:,} lignes chargées en mémoire ...")

        except Exception as e:
            print(f"⚠️  Erreur fichier {f}: {e}")

    if not chunks:
        print("❌ Aucune donnée chargée !")
        sys.exit(1)

    print(f"🔄 Concatenation de {total_rows_seen:,} lignes...")
    df = pd.concat(chunks, ignore_index=True)
    
    # Libérer la mémoire de la liste chunks
    del chunks 
    import gc
    gc.collect()

    print(f"\n{'='*60}")
    print(f"📊 STATISTIQUES (Sur {len(df):,} lignes chargées)")
    print(f"{'='*60}")
    
    # Optimisation : describe sur float32/int32 est plus rapide
    print(f"\n{df[['count', 'prediction']].describe()}")

    # Régression linéaire
    print(f"\n{'='*60}")
    print(f"📈 RÉGRESSION LINÉAIRE")
    
    # Scipy gère mal les float32 parfois, on convertit juste les colonnes nécessaires en numpy array pour le calcul
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["prediction"].values, df["count"].values
    )

    print(f"Équation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"R²: {r_value**2:.4f}")

    # Résidus (Calcul vectorisé rapide)
    residuals = df["count"] - df["prediction"]

    # Top archétypes
    print(f"\n{'='*60}")
    print(f"🏆 TOP 20 ARCHÉTYPES")
    
    top_archetypes = df.groupby("source")["count_source"].first().sort_values(ascending=False).head(20)
    for i, (arch, count) in enumerate(top_archetypes.items(), 1):
        print(f"  {i:2d}. {arch}: {count:,}")

    # Matrice de corrélation
    print(f"\n{'='*60}")
    print(f"🔗 CORRÉLATION")
    print(df[["count", "wins", "count_source", "count_target", "prediction"]].corr().to_string())

    # Sauvegardes
    print(f"\n{'='*60}")
    print(f"💾 SAUVEGARDE")
    
    sample_size = min(max_sample_size, len(df))
    df.sample(n=sample_size, random_state=42).to_csv(output_sample, index=False)
    print(f"✓ Échantillon viz ({sample_size}) -> {output_sample}")

    results = {
        "total_edges_analyzed": int(len(df)),
        "regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
        },
        "residuals": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
        },
        "top_archetypes": {k: int(v) for k, v in top_archetypes.items()}
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Stats JSON -> {output_json}")

if __name__ == "__main__":
    main()