# analyse_stats_light.py - Version LSD sans graphiques
import pandas as pd
import numpy as np
from scipy import stats
import glob
import sys
import os
import json
import argparse


def main():
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Analyse des statistiques de matchs d'archétypes"
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default="./stats_local",
        help="Chemin vers le répertoire contenant les fichiers stats (défaut: ./stats_local)",
    )
    parser.add_argument(
        "--output-sample",
        default="sample_for_viz.csv",
        help="Fichier de sortie pour l'échantillon (défaut: sample_for_viz.csv)",
    )
    parser.add_argument(
        "--output-json",
        default="stats_results.json",
        help="Fichier de sortie pour les statistiques JSON (défaut: stats_results.json)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Taille maximale de l'échantillon (défaut: 100000)",
    )

    args = parser.parse_args()

    stats_dir = args.stats_dir
    output_sample = args.output_sample
    output_json = args.output_json
    max_sample_size = args.sample_size

    print(f"📂 Lecture des fichiers depuis {stats_dir}...")

    # Vérifier que le répertoire existe
    if not os.path.exists(stats_dir):
        print(f"❌ Le répertoire {stats_dir} n'existe pas!")
        print("\nUtilisation:")
        print(
            f"  python3 {sys.argv[0]} <stats_dir> [--output-sample FILE] [--output-json FILE] [--sample-size N]"
        )
        print("\nExemple:")
        print(f"  python3 {sys.argv[0]} ./stats_local")
        print(f"  python3 {sys.argv[0]} /path/to/stats --sample-size 50000")
        sys.exit(1)

    # Lire tous les fichiers
    files = glob.glob(os.path.join(stats_dir, "part-*"))
    if not files:
        print(f"❌ Aucun fichier part-* trouvé dans {stats_dir}")
        sys.exit(1)

    print(f"📂 Lecture de {len(files)} fichier(s)...")

    # Lecture par chunks pour économiser la mémoire
    chunks = []
    for f in files:
        try:
            # Lire par petits morceaux
            for chunk in pd.read_csv(
                f,
                sep=";",
                names=[
                    "source",
                    "target",
                    "count",
                    "wins",
                    "count_source",
                    "count_target",
                    "prediction",
                ],
                chunksize=100000,  # Lire 100k lignes à la fois
            ):
                chunks.append(chunk)
        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture de {f}: {e}")

    if not chunks:
        print("❌ Aucune donnée chargée!")
        sys.exit(1)

    print("🔄 Concatenation des données...")
    df = pd.concat(chunks, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"📊 STATISTIQUES DESCRIPTIVES")
    print(f"{'='*60}")
    print(f"Total edges: {len(df):,}")
    print(f"Archétypes uniques (source): {df['source'].nunique():,}")
    print(f"Archétypes uniques (target): {df['target'].nunique():,}")
    print(f"\n{df[['count', 'prediction']].describe()}")

    # Régression linéaire
    print(f"\n{'='*60}")
    print(f"📈 RÉGRESSION LINÉAIRE")
    print(f"{'='*60}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["prediction"], df["count"]
    )

    print(f"Équation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"Coefficient de corrélation (r): {r_value:.4f}")
    print(f"Coefficient de détermination (R²): {r_value**2:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Erreur standard: {std_err:.4f}")

    # Résidus
    print(f"\n{'='*60}")
    print(f"📊 ANALYSE DES RÉSIDUS")
    print(f"{'='*60}")

    residuals = df["count"] - df["prediction"]
    print(f"Moyenne des résidus: {residuals.mean():.4f}")
    print(f"Écart-type des résidus: {residuals.std():.4f}")
    print(f"Médiane des résidus: {residuals.median():.4f}")
    print(f"Min résidu: {residuals.min():.4f}")
    print(f"Max résidu: {residuals.max():.4f}")

    # Top archétypes
    print(f"\n{'='*60}")
    print(f"🏆 TOP 20 ARCHÉTYPES LES PLUS JOUÉS")
    print(f"{'='*60}")

    top_archetypes = (
        df.groupby("source")["count_source"]
        .first()
        .sort_values(ascending=False)
        .head(20)
    )

    for i, (arch, count) in enumerate(top_archetypes.items(), 1):
        print(f"  {i:2d}. {arch}: {count:,} matchs")

    # Matrice de corrélation
    print(f"\n{'='*60}")
    print(f"🔗 MATRICE DE CORRÉLATION")
    print(f"{'='*60}")

    corr = df[
        ["count", "wins", "count_source", "count_target", "prediction"]
    ].corr()
    print(corr.to_string())

    # Sauvegarder les données pour visualisation ultérieure
    print(f"\n{'='*60}")
    print(f"💾 SAUVEGARDE DES DONNÉES POUR VISUALISATION")
    print(f"{'='*60}")

    # Échantillonner les données pour la visualisation
    sample_size = min(max_sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Sauvegarder en CSV léger
    df_sample.to_csv(output_sample, index=False)
    print(
        f"✓ {sample_size:,} points échantillonnés sauvegardés dans {output_sample}"
    )

    # Sauvegarder les statistiques en JSON
    results = {
        "total_edges": int(len(df)),
        "unique_source": int(df["source"].nunique()),
        "unique_target": int(df["target"].nunique()),
        "regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
        },
        "residuals": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "median": float(residuals.median()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        },
        "top_archetypes": {k: int(v) for k, v in top_archetypes.items()},
        "correlation_matrix": corr.to_dict(),
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Statistiques sauvegardées dans {output_json}")

    print(f"\n{'='*60}")
    print("✅ Analyse terminée avec succès!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
