import os
import numpy as np
import pandas as pd
import igraph as ig
from collections import Counter

DATA_DIR = "data/prepared"

# ── Modularity comparability settings ─────────────────────────────────────────
# Raw Louvain modularity (Q) is NOT directly comparable across platforms: the
# number of communities grows with network size and Q is subject to a resolution
# limit. Two comparability aids are provided:
#   (1) ALWAYS (cheap): n_communities is reported alongside Q, making the size
#       dependence explicit.
#   (2) OPTIONAL (expensive): a standardised z-score of Q against a
#       degree-preserving configuration-model null (edge rewiring):
#         z_Q = (Q_obs - mean(Q_null)) / sd(Q_null).
# The null requires rewiring the graph and re-running Louvain many times, which
# is very slow on large graphs (e.g. NPM), so it is OFF by default. Turn it on
# with COMPUTE_MODULARITY_NULL=True; it still auto-skips graphs above
# MODULARITY_NULL_MAX_EDGES unless FORCE_MODULARITY_NULL=True. Recommended use:
# enable it on a dedicated run for the smaller platforms (PyPI, Packagist).
COMPUTE_MODULARITY_NULL   = False
MODULARITY_NULL_REPLICAS  = 20
MODULARITY_NULL_MAX_EDGES = 200_000
FORCE_MODULARITY_NULL     = False


def modularity_null_zscore(g_u, membership, n_edges):
    """z-score of observed modularity vs a degree-preserving rewired null."""
    q_obs = g_u.modularity(membership)
    if not COMPUTE_MODULARITY_NULL:
        return q_obs, np.nan
    if n_edges > MODULARITY_NULL_MAX_EDGES and not FORCE_MODULARITY_NULL:
        print(f"    modularity null skipped: {n_edges:,} edges > "
              f"{MODULARITY_NULL_MAX_EDGES:,} (set FORCE_MODULARITY_NULL=True to run)")
        return q_obs, np.nan
    print(f"    modularity null: {MODULARITY_NULL_REPLICAS} rewired replicas "
          f"({n_edges:,} edges)...")
    q_null = []
    for r in range(MODULARITY_NULL_REPLICAS):
        gr = g_u.copy()
        gr.rewire(n=10 * gr.ecount())          # degree-preserving edge swaps
        q_null.append(gr.community_multilevel().modularity)
    q_null = np.array(q_null)
    sd = q_null.std(ddof=1)
    z  = (q_obs - q_null.mean()) / sd if sd > 0 else np.nan
    return q_obs, z


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def gini(arr):
    """Compute Gini coefficient of an array."""
    arr = np.array(arr, dtype=float)
    arr = arr[arr >= 0]
    if len(arr) == 0:
        return np.nan
    arr = np.sort(arr)
    n = len(arr)
    return (2 * np.sum(np.arange(1, n+1) * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


def power_law_exponent(degrees):
    """Estimate power law exponent via MLE (Clauset et al.)."""
    degrees = np.array([d for d in degrees if d > 0])
    if len(degrees) < 10:
        return np.nan
    xmin = 1
    n = len(degrees)
    return 1 + n * (np.sum(np.log(degrees / (xmin - 0.5)))) ** -1


def build_graph(nodes_df, edges_df, platform):
    """Build an igraph directed graph for a single platform."""
    n = nodes_df[nodes_df["platform"] == platform].copy().reset_index(drop=True)
    e = edges_df[edges_df["platform"] == platform].copy()

    # Map package names to integer indices
    name_to_idx = {name: i for i, name in enumerate(n["name"])}

    # Build edge list as integer pairs
    src = e["project_name"].map(name_to_idx)
    tgt = e["dependency_name"].map(name_to_idx)

    # Drop edges where either endpoint isn't in our node index
    mask = src.notna() & tgt.notna()
    src = src[mask].astype(int).tolist()
    tgt = tgt[mask].astype(int).tolist()

    g = ig.Graph(n=len(n), edges=list(zip(src, tgt)), directed=True)
    g.vs["name"] = n["name"].tolist()

    print(f"  {platform}: {g.vcount():,} nodes, {g.ecount():,} edges")
    return g, n, name_to_idx


def compute_node_metrics(g, platform):
    """Compute all node-level metrics for a graph."""
    print(f"  Computing metrics for {platform}...")

    metrics = pd.DataFrame({"name": g.vs["name"]})

    # Degree
    metrics["in_degree"]  = g.indegree()
    metrics["out_degree"] = g.outdegree()

    n = g.vcount()

    # PageRank (directed, damping=0.85)
    # NOTE (cross-platform comparability): PageRank sums to 1 across all nodes,
    # so the average node scores 1/n. Raw PageRank is therefore NOT comparable
    # across platforms of different size. We keep the raw score for within-
    # platform ranking, and add `pagerank_norm = pagerank * n`, which rescales
    # so that the average node = 1.0 and values read as "times the mean node's
    # centrality" — a size-invariant measure suitable for cross-platform tables.
    pr = g.pagerank(damping=0.85, directed=True)
    metrics["pagerank"]      = pr
    metrics["pagerank_norm"] = [p * n for p in pr]

    # Betweenness — expensive on large graphs, sample if needed.
    # NOTE (cross-platform comparability): raw betweenness counts the number of
    # shortest paths through a node and scales with the number of node pairs,
    # so raw values are NOT comparable across networks of different size. We
    # divide by the maximum possible pair count (n-1)(n-2) for a directed graph
    # to obtain the standard normalised betweenness in [0, 1].
    if n > 100_000:
        print(f"    Betweenness: graph too large ({n:,} nodes), using cutoff approximation...")
        # Distance-capped betweenness keeps this tractable; the same
        # normalisation constant is applied so platforms remain comparable.
        bw = g.betweenness(directed=True, cutoff=5)
    else:
        bw = g.betweenness(directed=True)
    bw_norm_factor = (n - 1) * (n - 2) if n > 2 else 1.0
    metrics["betweenness"] = [b / bw_norm_factor for b in bw]

    # Clustering coefficient (undirected version for interpretability)
    g_undirected = g.as_undirected(mode="collapse")
    metrics["clustering"] = g_undirected.transitivity_local_undirected(mode="zero")

    # K-core decomposition (on undirected graph)
    metrics["kcore"] = g_undirected.coreness()

    # Louvain community membership (computed once here on the undirected graph).
    # Stored per node so that (a) 03 can report modularity comparably and
    # (b) 06 can use community as a clustering variable for network-robust
    # standard errors (packages in the same community are not independent).
    communities = g_undirected.community_multilevel()
    metrics["community"] = communities.membership

    metrics["platform"] = platform
    return metrics


def compute_ecosystem_metrics(g, platform, node_metrics):
    """Compute ecosystem-level structural metrics."""
    print(f"  Ecosystem metrics for {platform}...")

    n_nodes = g.vcount()
    n_edges = g.ecount()

    # Density
    density = g.density()

    # Giant weakly connected component
    wcc = g.clusters(mode="weak")
    giant_wcc = max(wcc.sizes()) / n_nodes

    # Giant strongly connected component
    scc = g.clusters(mode="strong")
    giant_scc = max(scc.sizes()) / n_nodes

    # Gini of in-degree
    in_degrees = node_metrics["in_degree"].values
    gini_indegree = gini(in_degrees)

    # Power law exponent of in-degree
    pl_exp = power_law_exponent(in_degrees)

    # Average clustering
    avg_clustering = node_metrics["clustering"].mean()

    # K-core max
    kcore_max = node_metrics["kcore"].max()

    # Modularity via community detection (undirected).
    # Reuse the Louvain membership already computed at node level (avoids running
    # Louvain twice) and standardise Q against a configuration-model null so it
    # is comparable across platforms of different size.
    g_u        = g.as_undirected(mode="collapse")
    membership = node_metrics["community"].values
    comm_sizes = node_metrics["community"].value_counts()
    n_comms    = int(len(comm_sizes))              # includes isolated singletons
    n_comms_ge2 = int((comm_sizes >= 2).sum())     # meaningful multi-node communities
    modularity, modularity_z = modularity_null_zscore(g_u, membership, n_edges)

    # % of nodes in k-core >= 2 (the "core" of the ecosystem)
    pct_in_core = (node_metrics["kcore"] >= 2).mean()

    return {
        "platform":       platform,
        "n_nodes":        n_nodes,
        "n_edges":        n_edges,
        "density":        round(density, 8),
        "giant_wcc_pct":  round(giant_wcc * 100, 2),
        "giant_scc_pct":  round(giant_scc * 100, 2),
        "gini_indegree":  round(gini_indegree, 4),
        "power_law_exp":  round(pl_exp, 4),
        "avg_clustering": round(avg_clustering, 4),
        "kcore_max":      int(kcore_max),
        "modularity":     round(modularity, 4),
        "n_communities":  n_comms,
        "n_communities_ge2": n_comms_ge2,
        "modularity_z":   round(modularity_z, 4) if np.isfinite(modularity_z) else np.nan,
        "pct_in_core":    round(pct_in_core * 100, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load prepared data
    section("Loading prepared data")
    nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(DATA_DIR, "edges.parquet"))
    print(f"Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")

    all_node_metrics  = []
    all_eco_metrics   = []

    for platform in ["NPM", "Pypi", "Packagist"]:
        section(f"Platform: {platform}")

        # Build graph
        g, n_df, _ = build_graph(nodes, edges, platform)

        # Node metrics
        nm = compute_node_metrics(g, platform)
        all_node_metrics.append(nm)

        # Ecosystem metrics
        em = compute_ecosystem_metrics(g, platform, nm)
        all_eco_metrics.append(em)

        # Print top 10 packages by PageRank
        top10 = nm.nlargest(10, "pagerank")[["name", "pagerank", "in_degree", "kcore"]]
        print(f"\n  Top 10 by PageRank ({platform}):")
        print(top10.to_string(index=False))

    # ── Merge metrics back onto node table ────────────────────────────────────

    section("Merging metrics onto node table")

    metrics_df = pd.concat(all_node_metrics, ignore_index=True)
    nodes_final = nodes.merge(
        metrics_df[["platform", "name", "in_degree", "out_degree",
                    "pagerank", "pagerank_norm", "betweenness", "clustering",
                    "kcore", "community"]],
        on=["platform", "name"],
        how="left"
    )

    # Fill zeros for packages not in any edge (isolated nodes)
    for col in ["in_degree", "out_degree", "pagerank", "pagerank_norm",
                "betweenness", "clustering", "kcore"]:
        nodes_final[col] = nodes_final[col].fillna(0)
    # Isolated nodes form their own singleton community (label -1)
    nodes_final["community"] = nodes_final["community"].fillna(-1).astype(int)

    print(f"Final node table with metrics: {len(nodes_final):,} rows x {nodes_final.shape[1]} columns")

    # ── Save outputs ──────────────────────────────────────────────────────────

    section("Saving outputs")

    out_nodes = os.path.join(DATA_DIR, "nodes_with_metrics.parquet")
    out_eco   = os.path.join(DATA_DIR, "ecosystem_metrics.csv")

    nodes_final.to_parquet(out_nodes, index=False)
    print(f"Saved: nodes_with_metrics.parquet")

    eco_df = pd.DataFrame(all_eco_metrics)
    eco_df.to_csv(out_eco, index=False)
    print(f"\nEcosystem metrics:")
    print(eco_df.to_string(index=False))
    eco_df.to_csv(out_eco, index=False)

    print("\nGraph construction complete. Ready for analysis.")
