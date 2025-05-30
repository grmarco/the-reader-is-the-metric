import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob
import os
from matplotlib.ticker import MultipleLocator


def optimal_clusters(features, method="elbow", max_clusters=10):
    """
    Determines the optimal number of clusters using the selected method.
    """
    distortions = []
    silhouettes = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        distortions.append(kmeans.inertia_)  
        silhouettes.append(silhouette_score(features, cluster_labels)) 
    
    if method == "elbow":
        diff = np.diff(distortions)
        diff2 = np.diff(diff)
        optimal_k = np.argmax(diff2) + 2
    elif method == "silhouette":
        optimal_k = np.argmax(silhouettes) + 2  
    else:
        raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")
    return optimal_k

def filter_outliers(features, threshold=1.5):
    """
    Filters outliers using the interquartile range (IQR) method.
    Returns a boolean mask: True indicates that the point is NOT an outlier.
    """
    Q1 = np.percentile(features, 25, axis=0)
    Q3 = np.percentile(features, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    mask = np.all((features >= lower_bound) & (features <= upper_bound), axis=1)
    return mask

###############################################################################
# Funciones de visualización y reporte
###############################################################################
def plot_pcs_tsne_with_clusters(df, features, clustering_method="elbow", max_clusters=10, remove_outliers=True, dataset_title=""):
    """
    Perform PCA and visualise clusters with Convex Hulls.

    In addition:
      1. Authors whose name starts with "reader" are represented with the same colour and marker,
         are excluded from clustering (but are assigned a cluster with predict) and are indicated in the legend.
      2. Authors beginning with "human" are distinguished with a square marker.
      3. Title and axis labels are added.
      4. The legend is placed at the bottom, centred and spread across the width.
    """

    # 1. Separate the "readers" from the others
    readers_mask = df['author'].str.lower().str.startswith('reader')
    df_readers = df[readers_mask]
    df_others = df[~readers_mask]
    
    # Separate the corresponding features
    features_readers = features[readers_mask]
    features_others = features[~readers_mask]

    # 2. Determine the optimal number of clusters (excluding "reader") and apply KMeans
    n_clusters = optimal_clusters(features_others, method=clustering_method, max_clusters=max_clusters)
    print(f"Optimal number of clusters determined (excluding 'reader*'): {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_others)  # Ajuste solo con los "otros" autores

    # Tags for "other" authors and prediction for "reader*".
    cluster_labels_others = kmeans.labels_
    cluster_labels_readers = kmeans.predict(features_readers)

    # Combine labels in original order
    cluster_labels = np.empty(len(df), dtype=int)
    cluster_labels[~readers_mask] = cluster_labels_others
    cluster_labels[readers_mask] = cluster_labels_readers
    df["cluster"] = cluster_labels

    # 3. Dimensionality reduction with PCA (all points)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # 4. Filter outliers for display (if requested)
    inlier_mask_pca = filter_outliers(pca_result) if remove_outliers else np.ones(len(df), dtype=bool)

    # Preparing colour palettes for non-reader authors
    unique_authors_others = sorted(df_others['author'].unique())
    palette_cat = sns.color_palette("Set1", n_colors=len(unique_authors_others))
    color_dict = dict(zip(unique_authors_others, palette_cat))
    reader_color = "black"
    reader_marker = "x"

    # Create the figure (5x5) and draw convex hulls for each cluster.
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5, 5))
    for cluster in np.unique(cluster_labels):
        if remove_outliers:
            cluster_points = pca_result[(cluster_labels == cluster) & inlier_mask_pca]
        else:
            cluster_points = pca_result[cluster_labels == cluster]

        if len(cluster_points) > 2:
            try:
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices]
                ax.fill(hull_vertices[:, 0],
                        hull_vertices[:, 1],
                        alpha=0.2, label=f"Cluster {cluster}")
                for simplex in hull.simplices:
                    line_x = cluster_points[simplex, 0]
                    line_y = cluster_points[simplex, 1]
                    ax.plot(line_x, line_y, 'k-', linewidth=1.0)
            except Exception as e:
                print(f"Error en el cluster {cluster}: {e}")
                pass

    # 7. Graph each author's points
    # "Normal" authors (not "reader")
    for author in unique_authors_others:
        mask_author = (df['author'] == author)
        indices = mask_author & inlier_mask_pca
        marker = "s" if author.lower().startswith("human") else "o"
        ax.scatter(
            pca_result[indices, 0],
            pca_result[indices, 1],
            c=[color_dict[author]], marker=marker, s=80,
            edgecolor="black", linewidth=0.5,
            label=author
        )
    # Chart "reader*" with item in the legend
    mask_readers = readers_mask & inlier_mask_pca
    ax.scatter(
        pca_result[mask_readers, 0],
        pca_result[mask_readers, 1],
        c=reader_color, marker=reader_marker, s=80,
        edgecolor="black", linewidth=0.5,
        label="Ideal Reader Texts"
    )

    # 8. Tags and title (includes the title of the dataset)
    ax.set_title(f"PCA - Clusters for {dataset_title}", fontsize=12)
    ax.set_xlabel("Principal Component 1", fontsize=10)
    ax.set_ylabel("Principal Component 2", fontsize=10)

    # 9. Legend at the bottom, centred and extended
    handles, labels = ax.get_legend_handles_labels()
    unique_items = list(dict(zip(labels, handles)).items())
    new_labels, new_handles = zip(*unique_items)
    ax.legend(new_handles, new_labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.16),
              fancybox=True, shadow=True, ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"figures/texts_clusters/pca_clusters_{dataset_title}.pdf")
    plt.show()

    # 10. Display distribution of clusters by author
    cluster_distribution_by_dataset(df)

def cluster_distribution_by_dataset(df):
    """
    Analyses the number of elements in each cluster, separated by author.
    """
    cluster_counts = df.groupby(["cluster", "author"]).size().unstack(fill_value=0)
    print("\nCluster Distribution by Dataset:\n")
    print(cluster_counts)

def generate_latex_table_for_cluster(cluster_id, top_features, bottom_features, dataset_title=""):
    """
    Generates a LaTeX table showing the top and bottom features for a cluster.
    """
    table = "\\begin{table}[ht]\n\\centering\n"
    table += f"\\caption{{Dataset: {dataset_title} - Cluster {cluster_id}: Top and Bottom Features}}\n"
    table += "\\begin{tabular}{ll}\n"
    table += "\\hline\n"
    table += "Top Features (Δ) & Bottom Features (Δ) \\\\\n"
    table += "\\hline\n"
    
    max_len = max(len(top_features), len(bottom_features))
    top_list = list(top_features.items())
    bottom_list = list(bottom_features.items())
    
    for i in range(max_len):
        top_str = f"{top_list[i][0]} ({top_list[i][1]:+.4f})" if i < len(top_list) else ""
        bottom_str = f"{bottom_list[i][0]} ({bottom_list[i][1]:+.4f})" if i < len(bottom_list) else ""
        table += f"{top_str} & {bottom_str} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += f"\\label{{tab:cluster_{cluster_id}_features}}\n"
    table += "\\end{table}\n"
    return table

def plot_cluster_differences(cluster_id, diff_series, dataset_title=""):
    """
    Generates a horizontal bar chart showing the difference (Δ) for each feature (cluster mean - overall mean) for a given cluster.
    (cluster mean - global mean) for a given cluster.
    """
    plt.figure(figsize=(8, 6))
    diff_sorted = diff_series.sort_values()
    colors = ['red' if x < 0 else 'green' for x in diff_sorted]
    plt.barh(diff_sorted.index, diff_sorted.values, color=colors)
    plt.xlabel("Difference (Cluster Mean - Global Mean)")
    plt.title(f"Feature Differences for Cluster {cluster_id} - {dataset_title}")
    plt.tight_layout()
    plt.show()

def describe_clusters(df, feature_cols, cluster_col='cluster', top_k=5, dataset_title=""):
    """
    For each cluster, calculate the difference between the mean of each feature in the cluster and the overall mean,
    and print:
      - The top 'top_k' features (with positive Δ) and bottom 'top_k' features (with negative Δ).
      - A LaTeX table with these results.
      - A bar chart showing all the differences.
    """
    global_means = df[feature_cols].mean()
    
    for cluster_id in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster_id]
        cluster_means = subset[feature_cols].mean()
        diff = cluster_means - global_means
        diff_sorted = diff.sort_values(ascending=False)
        
        top_features = diff_sorted.head(top_k)
        bottom_features = diff_sorted.tail(top_k)
        
        print(f"\n=== Cluster {cluster_id} - {dataset_title} ===")
        print("Top features (above global mean):")
        for feat, value in top_features.items():
            print(f"  {feat}: Δ = {value:.4f}")
        print("Bottom features (below global mean):")
        for feat, value in bottom_features.items():
            print(f"  {feat}: Δ = {value:.4f}")
            
        # Generar y mostrar tabla en LaTeX
        latex_table = generate_latex_table_for_cluster(cluster_id, top_features, bottom_features, dataset_title=dataset_title)
        print("\nLaTeX Table:")
        print(latex_table)
        
        # Generar gráfico de barras horizontales
        plot_cluster_differences(cluster_id, diff, dataset_title=dataset_title)

###############################################################################
# maun function: run_analysis
###############################################################################
def run_analysis(df, features, feature_cols, remove_outliers=True, dataset_title=""):
    """
    Run the full analysis:
      1. visualise the clusters using PCA and convex hulls.
      2. Calculate feature differences for each cluster.
      3. Generate LaTeX tables and difference graphs.

    Parameters:
      df : pd.DataFrame with the data (must include the 'author' column).
      features : Array of features (in the same order as df).
      remove_outliers : Boolean to indicate if outliers are filtered out.
      dataset_title : Title of the dataset (appears in titles and descriptions).
    """

    df = df.copy()  
    features = features.copy()
    # Display clusters and assign the "cluster" column to df
    plot_pcs_tsne_with_clusters(df, features, clustering_method="elbow", max_clusters=4,
                                  remove_outliers=remove_outliers, dataset_title=dataset_title)
    



def plot_f1(base_path, model_str):
    # Configure Seaborn style for a publication-quality figure with increased font sizes
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.3)
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # ----------------------------
    # Load per-user datasets
    user_file_paths = glob.glob(f'{base_path}/by_reader/{model_str}/*.csv')
    user_dataframes = [pd.read_csv(file) for file in user_file_paths]

    # Extract f1_test values for each dataset (per user)
    f1_test_per_user = []
    dataset_labels = []

    for file_path, df in zip(user_file_paths, user_dataframes):
        df = df[df['user_id'].notna()]  # Exclude records without a user_id
        #df = df[df['f1_test'] > 0.55]
        f1_values = df['f1_test'].to_list()
        if f1_values:
            f1_test_per_user.append(f1_values)
            label = os.path.basename(file_path).replace('_model_results.csv', '')
            dataset_labels.append(label)

    # ----------------------------
    # Load baseline values (majority voting)
    baseline_file_paths = glob.glob(f'{base_path}/baseline/{model_str}/*.csv')
    baseline_scores = {}
    for file_path in baseline_file_paths:
        df_baseline = pd.read_csv(file_path)
        if not df_baseline.empty:
            f1_baseline = df_baseline.iloc[0]['f1_test']
            label_base = os.path.basename(file_path).replace('_model_results.csv', '')
            baseline_scores[label_base] = f1_baseline

    # ----------------------------
    # Create the boxplot and overlay the baseline scores with markers and legend
    if f1_test_per_user:
        # Reduced width for a narrower figure
        plt.figure(figsize=(6, 6))
        
        # Use a harmonious, pastel palette for the boxplot
        palette = sns.color_palette("pastel")
        ax = sns.boxplot(data=f1_test_per_user, palette=palette)
        
        # Plot the user means as crimson circles with a label added once for the legend
        user_means = [pd.Series(data).mean() for data in f1_test_per_user]
        for i, mean in enumerate(user_means):
            plt.scatter(i, mean, color='crimson', s=60, zorder=3,
                        label='User Mean' if i == 0 else "")
        
        # Overlay the baseline scores as steelblue crosses with a label added once for the legend
        for i, label in enumerate(dataset_labels):
            if label in baseline_scores:
                baseline_value = baseline_scores[label]
                plt.scatter(i, baseline_value-.05, color='steelblue', marker='X', s=100, zorder=4,
                            label='Baseline (majority voting)' if i == 0 else "")
            else:
                print(f"Baseline not found for dataset '{label}'")
        
        plt.title('Comparison of F1 Test Scores per Reader')
        plt.xlabel('Dataset')
        plt.ylabel('F1 Test Score')
        plt.xticks(ticks=range(len(dataset_labels)), labels=dataset_labels, rotation=45, ha='right')
        plt.ylim(0.0, None)
        
        # Set y-axis ticks every 0.05
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/f1_test_scores_{model_str}.pdf', format='pdf')
        plt.show()
    else:
        print("No data available to plot.")