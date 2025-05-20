import numpy as np
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt

# load data 
file_path = "data/facebook_combined.txt"
G = nx.read_edgelist(file_path)

def network_plot(G):

    # calculate node degrees
    degrees = dict(G.degree())
    node_sizes = [deg for deg in degrees.values()]  

    # quick plot 
    plt.figure(figsize=(6, 4))
    nx.draw(
        G,
        with_labels=False,  
        node_size=node_sizes,
        node_color='lightblue',
        edge_color='gray'
    )
    plt.title("Network Graph (Nodes Scaled by Degree)")
    plt.show()

## renormalization ##  
# steps: 
# 1. do the degree coarse graining and check clustering.
# 2. embed and check $beta$
network_plot(G)

# 1--degree threshold normalization
def degree_threshold_renormalization(G, thresholds):
    nested_graphs = {}
    for t in thresholds:
        # Select nodes with degree >= t
        nodes = [n for n, deg in G.degree() if deg >= t]
        subG = G.subgraph(nodes).copy()
        nested_graphs[t] = subG
    return nested_graphs

thresholds = list(range(200))
graphs = degree_threshold_renormalization(G, thresholds)

network_plot(graphs[30])

def compute_graph_stats(graphs_dict):
    """
    Compute stats for a dictionary of graphs indexed by threshold.
    """
    stats = []

    for t, G in graphs_dict.items():
        num_nodes = G.number_of_nodes()
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
        clustering = nx.average_clustering(G) if num_nodes > 0 else 0

        stats.append({
            'threshold': t,
            'num_nodes': num_nodes,
            'avg_degree': avg_degree,
            'avg_clustering': clustering,
            'degree_counts': {k: v for k, v in enumerate(nx.degree_histogram(G)) if v > 0}
        })

    return stats

# takes a minute 
g_stats = compute_graph_stats(graphs)

# plot some of it now 
thresholds = [s['threshold'] for s in g_stats]
num_nodes = [s['num_nodes'] for s in g_stats]
avg_degrees = [s['avg_degree'] for s in g_stats]
avg_clusterings = [s['avg_clustering'] for s in g_stats]
deg_counts = [s['degree_counts'] for s in g_stats]

# Plot 1: Number of nodes
def basic_plot(x, y, xlab="", ylab="", titel=""):

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titel)
    plt.grid(True)
    plt.show()

basic_plot(thresholds, num_nodes) # check loglog
basic_plot(thresholds, avg_degrees) # looks fine apparently.
basic_plot(thresholds, avg_clusterings) # not similar at all across.
# divide by average clustering per subgraph?
# not entirely sure what she meant. 

# NB: also do knn.

# find the peak 
# idx max something
def plot_degree_distribution(G, label=None):
    # Get degree sequence
    degrees = [d for _, d in G.degree()]
    degree_count = {}
    for d in degrees:
        degree_count[d] = degree_count.get(d, 0) + 1

    # Sort and normalize to get P(k)
    ks = sorted(degree_count.keys())
    pk = [degree_count[k] / sum(degree_count.values()) for k in ks]

    # Plot log-log
    plt.figure(figsize=(5, 4))
    plt.loglog(ks, pk, marker='o', linestyle='', label=label or "Data")
    plt.xlabel(r"$k$", fontsize=14)
    plt.ylabel(r"$P(k)$", fontsize=14)
    plt.title("Degree Distribution (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_degree_distribution(graphs[0], label="Treshold = 0")
plot_degree_distribution(graphs[3], label="Threshold = 3")
plot_degree_distribution(graphs[10], label="Threshold = 10")

# first do exponential binning 
def log_binned_degree_distribution(G, bins=20, label=None):
    degrees = np.array([d for _, d in G.degree()])
    degree_count = np.bincount(degrees)
    ks = np.nonzero(degree_count)[0]
    pk = degree_count[ks] / sum(degree_count)

    # Log-binning
    min_k = max(ks.min(), 1)
    max_k = ks.max()
    log_bins = np.logspace(np.log10(min_k), np.log10(max_k), bins)

    binned_ks = []
    binned_pk = []

    for i in range(len(log_bins) - 1):
        k_min, k_max = log_bins[i], log_bins[i+1]
        mask = (ks >= k_min) & (ks < k_max)
        if np.any(mask):
            k_avg = np.mean(ks[mask])
            pk_avg = np.mean(pk[mask])
            binned_ks.append(k_avg)
            binned_pk.append(pk_avg)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.loglog(binned_ks, binned_pk, 'o', label=label or "Log-binned")
    plt.xlabel(r"$k$", fontsize=14)
    plt.ylabel(r"$P(k)$", fontsize=14)
    plt.title("Log-binned Degree Distribution")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()

# still not sure about this.
log_binned_degree_distribution(G, bins=30)

# instead of p(k) raw do cumulative 
# complementary cumulative distribution function (CCDF)
def plot_degree_ccdf(G, label=None):
    degrees = np.array([d for _, d in G.degree()])
    max_k = degrees.max()

    # Degree counts
    values, counts = np.unique(degrees, return_counts=True)
    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    counts = counts[sorted_idx]

    # Normalize to get P(k)
    pk = counts / counts.sum()

    # Compute CCDF
    ccdf = 1 - np.cumsum(pk)

    # Shift values so P(k ≥ k) not P(k > k)
    values = values[:-1]
    ccdf = ccdf[:-1]

    # Plot
    plt.figure(figsize=(7, 5))
    plt.loglog(values, ccdf, marker='o', linestyle='', label=label or "CCDF")
    plt.xlabel(r"$k$", fontsize=14)
    plt.ylabel(r"$P(k \geq k)$", fontsize=14)
    plt.title("Degree CCDF (Cumulative Distribution)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()

# interesting, hmmm
plot_degree_ccdf(G)