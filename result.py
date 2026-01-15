# visualisation_robust.py
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
import seaborn as sns  # Optionnel mais rend la matrice plus jolie

def main():
    # Vérification des fichiers
    if not os.path.exists("sample_for_viz.csv") or not os.path.exists("stats_results.json"):
        print("❌ Fichiers manquants : Assurez-vous d'avoir exécuté l'analyse avant.")
        sys.exit(1)

    # Charger les données échantillonnées
    print("📂 Chargement des données échantillonnées...")
    df = pd.read_csv("sample_for_viz.csv")

    # Charger les statistiques JSON
    with open("stats_results.json", "r") as f:
        results = json.load(f)

    print(f"Données chargées: {len(df):,} points")

    # Création du style global
    plt.style.use('ggplot') 

    # 1. Nuage de points avec régression
    print("📊 Génération : observed_vs_predicted.png")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot léger (alpha bas pour voir la densité)
    plt.scatter(
        df["prediction"], df["count"], alpha=0.3, s=5, color="steelblue", label="Données (échantillon)"
    )

    # Ligne de régression (depuis le JSON)
    if "regression" in results:
        reg = results["regression"]
        # On crée des points X triés pour tracer une belle ligne
        x_line = pd.Series([df["prediction"].min(), df["prediction"].max()])
        y_line = reg["slope"] * x_line + reg["intercept"]
        
        r2_label = f"$R^2$={reg['r_squared']:.4f}" if 'r_squared' in reg else ""
        
        plt.plot(
            x_line,
            y_line,
            "r",
            linewidth=2,
            label=f"Régression (Analyse complète)\ny={reg['slope']:.2f}x+{reg['intercept']:.2f}\n{r2_label}",
        )

    plt.xlabel("Probabilité prédite", fontsize=12)
    plt.ylabel("Nombre de matchs réel", fontsize=12)
    plt.title("Matchs observés vs prédits", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig("observed_vs_predicted.png", dpi=300, bbox_inches="tight")

    # 2. Distribution des résidus
    print("📊 Génération : residuals_distribution.png")
    plt.figure(figsize=(10, 6))
    # Recalcul local des résidus pour être sûr
    residuals = df["count"] - df["prediction"]
    
    plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="coral", density=True)
    
    # Ajout d'une ligne KDE (densité) pour voir la forme
    try:
        residuals.plot.kde(color="darkred", linewidth=2, label="Densité")
    except:
        pass # Si scipy manque, on ignore

    plt.xlabel("Résidu (Réel - Prédit)", fontsize=12)
    plt.ylabel("Fréquence (Densité)", fontsize=12)
    plt.title("Distribution des erreurs de prédiction", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Zéro (Parfait)")
    plt.legend()
    plt.savefig("residuals_distribution.png", dpi=300, bbox_inches="tight")

    # 3. Top archétypes
    print("📊 Génération : top_archetypes.png")
    if "top_archetypes" in results:
        plt.figure(figsize=(12, 6))
        top_arch = results["top_archetypes"]
        # Trier par valeur décroissante pour être sûr
        sorted_archs = sorted(top_arch.items(), key=lambda x: x[1], reverse=True)
        archs = [x[0] for x in sorted_archs]
        counts = [x[1] for x in sorted_archs]

        plt.bar(range(len(archs)), counts, color="seagreen", edgecolor="black", alpha=0.8)
        plt.xlabel("Archétype", fontsize=12)
        plt.ylabel("Nombre de matchs (Dataset complet)", fontsize=12)
        plt.title("Top 20 archétypes les plus joués", fontsize=14, fontweight="bold")
        plt.xticks(range(len(archs)), archs, rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig("top_archetypes.png", dpi=300, bbox_inches="tight")

    # 4. Matrice de corrélation (Recalculée localement pour robustesse)
    print("📊 Génération : correlation_matrix.png")
    plt.figure(figsize=(10, 8))
    
    # On sélectionne uniquement les colonnes numériques pertinentes
    cols_to_corr = ["count", "wins", "count_source", "count_target", "prediction"]
    # On filtre pour ne garder que celles qui existent vraiment dans le CSV
    cols_present = [c for c in cols_to_corr if c in df.columns]
    
    corr_df = df[cols_present].corr()

    # Utilisation de Seaborn si dispo (plus joli), sinon Matplotlib standard
    try:
        import seaborn as sns
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", square=True)
    except ImportError:
        # Fallback Matplotlib pur (ton code original)
        im = plt.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im)
        for i in range(len(corr_df)):
            for j in range(len(corr_df.columns)):
                plt.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", color="black")
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr_df.columns)), corr_df.columns)

    plt.title("Matrice de corrélation (Échantillon)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")

    print("\n✅ Tous les graphiques ont été générés avec succès !")

if __name__ == "__main__":
    main()