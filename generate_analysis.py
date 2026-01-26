import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.colors import LogNorm
import sys
import os
import argparse

def main():
    # 1. Gestion des arguments
    parser = argparse.ArgumentParser(description="Génère le graphique d'analyse (Mesure vs Estimation) à partir du CSV Hadoop.")
    
    # Argument 1 : Fichier d'entrée (CSV)
    parser.add_argument("file_path", nargs='?', default="stats_final.csv", 
                        help="Chemin vers le fichier CSV d'entrée (défaut: stats_final.csv)")
    
    # Argument 2 : Fichier de sortie (Image) - NOUVEAU
    parser.add_argument("output_path", nargs='?', default=None, 
                        help="Nom du fichier image de sortie (optionnel, défaut: auto-généré)")
    
    # Argument optionnel : Échantillonnage
    parser.add_argument("--sample", type=int, default=50000, 
                        help="Nombre de points à afficher sur le graphique (défaut: 50000)")
    
    args = parser.parse_args()
    FILE_PATH = args.file_path
    SAMPLE_SIZE = args.sample
    
    print(f"--- DÉBUT DE L'ANALYSE ---")
    
    # 2. Chargement des données
    if not os.path.exists(FILE_PATH):
        print(f"❌ Erreur : Le fichier '{FILE_PATH}' est introuvable.")
        print("   -> Assurez-vous d'avoir fait le 'getmerge' sur la gateway et le 'scp' vers votre PC.")
        sys.exit(1)

    print(f"📂 Chargement de {FILE_PATH}...")
    
    # Définition des noms de colonnes selon ton Stats.java
    # Format: source;target;count;wins;count_source;count_target;prediction
    col_names = ["source", "target", "observed", "wins", "total_source", "total_target", "predicted"]
    
    try:
        df = pd.read_csv(FILE_PATH, sep=";", names=col_names)
    except Exception as e:
        print(f"❌ Erreur de lecture CSV : {e}")
        sys.exit(1)

    print(f"✅ {len(df):,} lignes chargées.")

    # 3. Nettoyage et Calculs
    # On retire les cas pathologiques (division par zéro ou prediction nulle)
    df = df[df["predicted"] > 0].copy()
    
    # Calcul du Ratio (pour la couleur) : Observation / Prévision
    # Ratio = 1 : L'archétype apparaît exactement comme prévu par le hasard.
    # Ratio > 1 : Synergie (joués ensemble plus souvent que prévu).
    # Ratio < 1 : Anti-synergie.
    df["ratio"] = df["observed"] / df["predicted"]

    # 4. Régression Linéaire (Calculée sur TOUT le dataset, pas l'échantillon)
    print("🧮 Calcul de la régression linéaire...")
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["observed"], df["predicted"])
    
    line_eq = f"y = {slope:.4f}x + {intercept:.4f}"
    stats_text = (f"{line_eq}\n"
                  f"R² = {r_value**2:.4f}\n"
                  f"p-value = {p_value:.2e}")
    
    print(f"   -> Résultat : {line_eq}")

    # 5. Échantillonnage pour le visuel
    # Si le fichier est trop gros, le graphique sera un bloc noir illisible.
    # On prend un échantillon aléatoire pour le scatter plot, mais la ligne rouge reste vraie pour tout le monde.
    if len(df) > SAMPLE_SIZE:
        print(f"🎨 Échantillonnage de {SAMPLE_SIZE} points pour le graphique...")
        plot_data = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        plot_data = df

    # 6. Création du Graphique (Style Matplotlib Pro)
    plt.figure(figsize=(10, 7))
    
    # Scatter Plot
    # c=df['ratio'] permet de colorer selon la "force" du lien
    # norm=LogNorm() permet de mieux voir les variations de couleurs si les ratios explosent
    plt.scatter(
        plot_data["observed"], 
        plot_data["predicted"], 
        c=plot_data["ratio"], 
        cmap='viridis', 
        s=10, 
        alpha=0.6, 
        norm=LogNorm(), # Échelle logarithmique pour la couleur (optionnel, retirer si erreur)
        label='Données (Paires d\'archétypes)'
    )
    
    # Barre de couleur
    cbar = plt.colorbar()
    cbar.set_label('Ratio (Observé / Prédit)', rotation=270, labelpad=15)

    # Ligne de régression
    # On crée des points X min et max pour tracer la ligne
    x_range = np.array([df["observed"].min(), df["observed"].max()])
    y_range = slope * x_range + intercept
    
    plt.plot(x_range, y_range, color='red', linewidth=2, label='Régression Linéaire')

    # Boîte de texte (Statistiques)
    # Positionnée en haut à gauche (x=0.05, y=0.95 en coordonnées relatives)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Labels et Titre
    plt.xlabel('Mesure (Nombre de matchs observés)', fontsize=12)
    plt.ylabel('Estimation (Nombre de matchs prédits)', fontsize=12)
    plt.title('Validation : Estimation vs Mesure', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Sauvegarde
    if args.output_path:
        output_img = args.output_path
    else:
        # Nom automatique basé sur le fichier d'entrée
        base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
        output_img = f"analyse_{base_name}.png"
    
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé sous : {output_img}")
    
    # Affichage (si tu es sur ton PC local)
    # plt.show() 

if __name__ == "__main__":
    main()