# visualisation.py - Version locale avec graphiques
import pandas as pd
import matplotlib.pyplot as plt
import json


def main():
    # Charger les données échantillonnées
    print("📂 Chargement des données échantillonnées...")
    df = pd.read_csv("sample_for_viz.csv")

    # Charger les statistiques
    with open("stats_results.json", "r") as f:
        results = json.load(f)

    print(f"Données chargées: {len(df):,} points")

    # 1. Nuage de points avec régression
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["prediction"], df["count"], alpha=0.5, s=10, color="steelblue"
    )

    reg = results["regression"]
    x_line = df["prediction"].sort_values()
    y_line = reg["slope"] * x_line + reg["intercept"]
    plt.plot(
        x_line,
        y_line,
        "r",
        linewidth=2,
        label=f"y={reg['slope']:.3f}x+{reg['intercept']:.3f}\n$R^2$={reg['r_squared']:.4f}",
    )

    plt.xlabel("Prédiction (valeur attendue)", fontsize=12)
    plt.ylabel("Count (valeur observée)", fontsize=12)
    plt.title(
        "Matchs observés vs prédits (échantillon)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig("observed_vs_predicted.png", dpi=300, bbox_inches="tight")
    print("✓ observed_vs_predicted.png")

    # 2. Distribution des résidus
    plt.figure(figsize=(10, 6))
    residuals = df["count"] - df["prediction"]
    plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="coral")
    plt.xlabel("Résidu (observé - prédit)", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    plt.title("Distribution des résidus", fontsize=14, fontweight="bold")
    plt.axvline(
        x=0, color="r", linestyle="--", linewidth=2, label="Résidu = 0"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig("residuals_distribution.png", dpi=300, bbox_inches="tight")
    print("✓ residuals_distribution.png")

    # 3. Top archétypes
    plt.figure(figsize=(12, 6))
    top_arch = results["top_archetypes"]
    archs = list(top_arch.keys())
    counts = list(top_arch.values())

    plt.bar(range(len(archs)), counts, color="seagreen", edgecolor="black")
    plt.xlabel("Archétype", fontsize=12)
    plt.ylabel("Nombre total de matchs", fontsize=12)
    plt.title(
        "Top 20 archétypes les plus joués", fontsize=14, fontweight="bold"
    )
    plt.xticks(range(len(archs)), archs, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("top_archetypes.png", dpi=300, bbox_inches="tight")
    print("✓ top_archetypes.png")

    # 4. Matrice de corrélation
    plt.figure(figsize=(10, 8))
    corr_dict = results["correlation_matrix"]
    corr_df = pd.DataFrame(corr_dict)

    im = plt.imshow(
        corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    )
    plt.colorbar(im)

    for i in range(len(corr_df)):
        for j in range(len(corr_df.columns)):
            plt.text(
                j,
                i,
                f"{corr_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.xticks(
        range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right"
    )
    plt.yticks(range(len(corr_df.columns)), corr_df.columns)
    plt.title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")
    print("✓ correlation_matrix.png")

    print("\n✅ Tous les graphiques générés!")


if __name__ == "__main__":
    main()
