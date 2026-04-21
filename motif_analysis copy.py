import os
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from typing import Any

from utils import load_moltbook_graph, load_reddit_graph, get_platform_colors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_triadic_census(G):
    """
    Computes the frequency of all 16 directed 3-node motifs.
    Uses NetworkX's standard triadic census implementation.
    """
    logging.info(f"Computing triadic census for {G.name} ({G.number_of_nodes()} nodes)...")
    census = nx.triadic_census(G)
    return census

def compute_motif_percentages(real_census):
    """
    Computes the percentage that each triadic motif consumes of the total distribution.
    Excludes disconnected motifs (003, 012, 102).
    """
    # Filter out motifs where nodes are not part of a connected triad
    filtered_census = {m: c for m, c in real_census.items() if m not in ['003', '012', '102']}
    total_motifs = sum(real_census.values())
    metrics = {}
    
    for motif, count in filtered_census.items():
        percentage = (count / total_motifs * 100) if total_motifs > 0 else 0
        metrics[motif] = {
            "count": count,
            "percentage": float(percentage)
        }
    return metrics

def generate_null_models(G, n_models=30):
    """
    Generates random graph null models (Configuration Model) preserving in/out degree sequences.
    Filters out unsupported self-loops and parallel edges for consistency.
    """
    logging.info(f"Generating {n_models} null models for {G.name}...")
    in_seq = [d for n, d in G.in_degree()]
    out_seq = [d for n, d in G.out_degree()]
    
    censuses = []
    for i in range(n_models):
        random_G_multi = nx.directed_configuration_model(in_degree_sequence=in_seq, out_degree_sequence=out_seq, create_using=nx.MultiDiGraph)
        # Convert to DiGraph and remove self-loops
        random_G = nx.DiGraph(random_G_multi)
        random_G.remove_edges_from(nx.selfloop_edges(random_G))
        random_G.name = f"{G.name}_null_{i}"
        
        censuses.append(compute_triadic_census(random_G))
        if (i+1) % 10 == 0:
            logging.info(f"  Finished {i+1}/{n_models} null models")
            
    return censuses

def compute_motif_z_scores(real_census, null_censuses):
    """
    Computes Z-scores and Significance Profile (SP) for each motif type.
    Excludes disconnected motifs (003, 012, 102).
    """
    
    # Filter out motifs where nodes are not part of a connected triad
    filtered_census = {m: c for m, c in real_census.items() if m not in ['003', '012', '102']}
    motifs = list(filtered_census.keys())
    null_df = pd.DataFrame(null_censuses)
    
    metrics = {}
    sp_vector = []
    
    for motif in motifs:
        observed = real_census[motif]
        null_mean = null_df[motif].mean()
        null_std = null_df[motif].std()
        
        # Avoid division by zero
        if null_std == 0:
            if observed == null_mean:
                z_score = 0
            else:
                z_score = np.inf if observed > null_mean else -np.inf
        else:
            z_score = (observed - null_mean) / null_std
            
        metrics[motif] = {
            "observed": observed,
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score) if not np.isinf(z_score) else (999.0 if z_score > 0 else -999.0)
        }
        sp_vector.append(z_score if not np.isinf(z_score) else (999.0 if z_score > 0 else -999.0))
        
    # Motif Significance Profile (SP) normalizes Z-scores to unit length
    sp_vector = np.array(sp_vector)
    norm = np.linalg.norm(sp_vector)
    sp_normalized = (sp_vector / norm).tolist() if norm > 0 else [0] * len(motifs)
    
    for i, motif in enumerate(motifs):
        metrics[motif]["sp_score"] = float(sp_normalized[i])
        
    return metrics

def local_clustering_comparison(G_moltbook, G_reddit):
    """
    Computes local triangle participation (undirected clustering coefficient) 
    and tests for distribution differences using Mann-Whitney U.
    """
    logging.info("Computing local clustering distributions...")
    
    m_undirected = G_moltbook.to_undirected()
    r_undirected = G_reddit.to_undirected()
    
    m_clustering = list(nx.clustering(m_undirected).values())
    r_clustering = list(nx.clustering(r_undirected).values())
    
    logging.info("Running Mann-Whitney U test on clustering distributions...")
    stat, p_val = mannwhitneyu(m_clustering, r_clustering, alternative='two-sided')
    
    return {
        "mann_whitney_u_statistic": float(stat),
        "p_value": float(p_val),
        "moltbook_mean_clustering": float(np.mean(m_clustering)),
        "reddit_mean_clustering": float(np.mean(r_clustering))
    }

def plot_motif_percentages(molt_metrics, reddit_metrics):
    """
    Generates a publication-quality motif percentage distribution plot.
    """
    plot_motifs = list(molt_metrics.keys())
    
    m_scores = [molt_metrics[m]["percentage"] for m in plot_motifs]
    r_scores = [reddit_metrics[m]["percentage"] for m in plot_motifs]
    
    colors = get_platform_colors()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_motifs))
    width = 0.35
    
    plt.bar(x - width/2, m_scores, width, label='Moltbook', color=colors["Moltbook"], alpha=0.8)
    plt.bar(x + width/2, r_scores, width, label='Reddit', color=colors["Reddit"], alpha=0.8)
    
    plt.xlabel('Directed Triadic Motif Type', fontsize=12)
    plt.ylabel('Percentage of Total Motifs (%)', fontsize=12)
    plt.title('Motif Distribution: Moltbook vs Reddit', fontsize=14)
    plt.xticks(x, plot_motifs, rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig("outputs/motif_percentages.png", dpi=300)
    logging.info("Saved plot to motif_percentages.png")

def plot_significance_profile(molt_metrics, reddit_metrics, sp=False):
    """
    Generates a publication-quality Z-score profile plot.
    """
    motifs = list(molt_metrics.keys())
    plot_motifs = motifs
    
    metric_key = "z_score"
    y_label = "Z-score"
    if sp:
        metric_key = "sp_score"
        y_label = "Normalized Significance Profile (SP)"

    m_scores = [molt_metrics[m][metric_key] for m in plot_motifs]
    r_scores = [reddit_metrics[m][metric_key] for m in plot_motifs]
    
    colors = get_platform_colors()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_motifs))
    width = 0.35
    
    plt.bar(x - width/2, m_scores, width, label='Moltbook', color=colors["Moltbook"], alpha=0.8)
    plt.bar(x + width/2, r_scores, width, label='Reddit', color=colors["Reddit"], alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=1)
    # Highlight significant thresholds (Z = +/- 2)
    plt.axhline(2, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-2, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Directed Triadic Motif Type', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'Motif Significance Profile: {y_label}', fontsize=14)
    plt.xticks(x, plot_motifs, rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f"outputs/motif_{metric_key}.png", dpi=300)
    logging.info(f"Saved plot to motif_{metric_key}.png")

def run_motif_analysis():
    G_moltbook = load_moltbook_graph()
    G_reddit = load_reddit_graph()
    
    print("\n" + "="*50)
    print("Starting Motif Analysis (Section 4.2)")
    print("="*50)
    
    results: dict[str, Any] = {}
    
    for name, G in [("Moltbook", G_moltbook), ("Reddit", G_reddit)]:
        print(f"\nProcessing {name}...")
        real_census = compute_triadic_census(G)
        
        # Calculate percentages
        percentage_metrics = compute_motif_percentages(real_census)
        
        # Calculate Z-scores
        null_censuses = generate_null_models(G, n_models=30)
        z_score_metrics = compute_motif_z_scores(real_census, null_censuses)
        
        # Combine metrics
        combined_metrics = {}
        for motif in z_score_metrics.keys():
            combined_metrics[motif] = {**z_score_metrics[motif], "percentage": percentage_metrics.get(motif, {}).get("percentage", 0)}
        
        results[name] = combined_metrics
        
    print("\nComputing local density statistical tests...")
    stat_results = local_clustering_comparison(G_moltbook, G_reddit)
    results["statistical_tests"] = stat_results
    
    print("\nGenerating figures and exporting...")
    plot_motif_percentages(results["Moltbook"], results["Reddit"])
    plot_significance_profile(results["Moltbook"], results["Reddit"])
    plot_significance_profile(results["Moltbook"], results["Reddit"], sp=True)
    
    with open("outputs/motif_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved numeric results to motif_results.json")
    
    print("\n=== Analysis Summary ===")
    print(f"Moltbook Triad Density (mean local clustering): {stat_results['moltbook_mean_clustering']:.4f}")
    print(f"Reddit Triad Density (mean local clustering):   {stat_results['reddit_mean_clustering']:.4f}")
    print(f"Mann-Whitney U p-value (difference in clustering): {stat_results['p_value']:.4e}")
    print("========================\n")

if __name__ == "__main__":
    run_motif_analysis()
