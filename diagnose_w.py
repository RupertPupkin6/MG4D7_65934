import os
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

DATA_DIR  = "data/prepared"
PLATFORMS = ["NPM", "Pypi", "Packagist"]

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── Load data ─────────────────────────────────────────────────────────────────

section("Loading data")
nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_utility.parquet"))
edges = pd.read_parquet(os.path.join(DATA_DIR, "edges.parquet"))
print(f"Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")

all_diag = []

for plat in PLATFORMS:
    section(f"Platform: {plat}")

    n = nodes[nodes["platform"] == plat].reset_index(drop=True)
    e = edges[edges["platform"] == plat]

    name_to_idx = {name: i for i, name in enumerate(n["name"])}
    sz = len(n)

    src = e["project_name"].map(name_to_idx).dropna().astype(int)
    tgt = e["dependency_name"].map(name_to_idx).dropna().astype(int)
    valid = src.index.intersection(tgt.index)
    src_v = src.loc[valid].values
    tgt_v = tgt.loc[valid].values

    utility = n["utility_index"].fillna(0).values

    # ── W1: Original outgoing dependency W ────────────────────────────────────

    W_out = sparse.csr_matrix(
        (np.ones(len(src_v)), (src_v, tgt_v)), shape=(sz, sz))
    row_sums = np.array(W_out.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    W_out = sparse.diags(1.0 / row_sums) @ W_out

    Wy_out = np.array(W_out @ utility).flatten()
    has_nb_out = np.diff(W_out.indptr) > 0
    corr_out, _ = spearmanr(utility[has_nb_out], Wy_out[has_nb_out])

    print(f"\n  W_out (original outgoing dependency):")
    print(f"    Non-zeros:    {W_out.nnz:>10,}")
    print(f"    Density:      {W_out.nnz/(sz**2):>10.2e}")
    print(f"    Spatial corr: {corr_out:>10.4f}")

    # ── W2: Co-dependency W ────────────────────────────────────────────────────
    #
    # Build incidence matrix A where A[chooser, dependency] = 1
    # Then C = A.T @ A gives co-dependency counts:
    # C[i,j] = number of packages that depend on both i and j
    # Diagonal removed, then row-normalised.
    #
    # This captures: packages that developers tend to adopt together
    # — the natural "neighbourhood" for adoption choice modelling.

    print(f"\n  Building co-dependency W (A.T @ A)...")

    # Map choosers (project_name) to row indices
    chooser_idx = e["project_name"].map(name_to_idx).dropna().astype(int)
    dep_idx     = e["dependency_name"].map(name_to_idx).dropna().astype(int)
    valid2      = chooser_idx.index.intersection(dep_idx.index)
    ch_v        = chooser_idx.loc[valid2].values
    dp_v        = dep_idx.loc[valid2].values

    # A: chooser × dependency incidence matrix
    # shape: (sz, sz) — both choosers and dependencies are packages
    A = sparse.csr_matrix(
        (np.ones(len(ch_v)), (ch_v, dp_v)), shape=(sz, sz))

    # C = A.T @ A: dependency × dependency co-occurrence
    # C[i,j] = how many packages depend on both i and j
    C = (A.T @ A).tocsr()

    # Remove diagonal (self co-occurrence)
    C = C - sparse.diags(C.diagonal())

    # Only keep top-k co-occurrences per row to maintain sparsity
    # k calibrated from connected node count
    connected = (np.diff(W_out.indptr) > 0).sum()
    k_cooc = max(5, min(50, int(connected * 0.001)))
    print(f"    k for co-occurrence sparsification: {k_cooc}")

    # Sparsify: keep top-k per row
    C_sparse_rows, C_sparse_cols, C_sparse_vals = [], [], []
    C_csr = C.tocsr()
    for i in range(sz):
        row_start = C_csr.indptr[i]
        row_end   = C_csr.indptr[i+1]
        if row_end == row_start:
            continue
        row_cols = C_csr.indices[row_start:row_end]
        row_vals = C_csr.data[row_start:row_end]
        if len(row_vals) > k_cooc:
            top_k_idx = np.argpartition(row_vals, -k_cooc)[-k_cooc:]
            row_cols  = row_cols[top_k_idx]
            row_vals  = row_vals[top_k_idx]
        C_sparse_rows.extend([i] * len(row_cols))
        C_sparse_cols.extend(row_cols.tolist())
        C_sparse_vals.extend(row_vals.tolist())

    W_cooc = sparse.csr_matrix(
        (C_sparse_vals, (C_sparse_rows, C_sparse_cols)), shape=(sz, sz))

    # Row-normalise
    row_sums_c = np.array(W_cooc.sum(axis=1)).flatten()
    row_sums_c[row_sums_c == 0] = 1
    W_cooc = sparse.diags(1.0 / row_sums_c) @ W_cooc

    Wy_cooc = np.array(W_cooc @ utility).flatten()
    has_nb_cooc = np.diff(W_cooc.indptr) > 0
    corr_cooc, _ = spearmanr(utility[has_nb_cooc], Wy_cooc[has_nb_cooc])

    w_vals_cooc = W_cooc.data
    gini_cooc = 0.0
    if len(w_vals_cooc) > 0:
        sw = np.sort(w_vals_cooc)
        nw = len(sw)
        gini_cooc = (2*np.sum(np.arange(1,nw+1)*sw) -
                     (nw+1)*np.sum(sw)) / (nw*np.sum(sw))

    print(f"\n  W_cooc (co-dependency similarity):")
    print(f"    Non-zeros:        {W_cooc.nnz:>10,}")
    print(f"    Density:          {W_cooc.nnz/(sz**2):>10.2e}")
    print(f"    Nodes with edges: {has_nb_cooc.sum():>10,} "
          f"({has_nb_cooc.sum()/sz*100:.1f}%)")
    print(f"    Weight mean:      {w_vals_cooc.mean() if len(w_vals_cooc)>0 else 0:>10.6f}")
    print(f"    Gini of weights:  {gini_cooc:>10.4f}")
    print(f"    Spatial corr:     {corr_cooc:>10.4f}")

    # ── W3: Naive degree-based W ───────────────────────────────────────────────

    deg = n["in_degree"].fillna(0).values
    k_naive = max(5, min(20, int(connected * 0.0001)))
    top_k = np.argsort(deg)[-k_naive:]

    rows_n, cols_n, vals_n = [], [], []
    for i in range(sz):
        w = deg[top_k]
        s = w.sum()
        if s > 0:
            for j, v in zip(top_k, w/s):
                rows_n.append(i)
                cols_n.append(j)
                vals_n.append(v)

    W_naive = sparse.csr_matrix((vals_n, (rows_n, cols_n)), shape=(sz, sz))
    Wy_naive = np.array(W_naive @ utility).flatten()
    has_nb_naive = np.diff(W_naive.indptr) > 0
    corr_naive, _ = spearmanr(utility[has_nb_naive], Wy_naive[has_nb_naive])

    print(f"\n  W_naive (degree-based):")
    print(f"    Non-zeros:    {W_naive.nnz:>10,}")
    print(f"    Spatial corr: {corr_naive:>10.4f}")

    # ── Comparison ────────────────────────────────────────────────────────────

    print(f"\n  ── Comparison summary ──")
    print(f"    W_out  spatial corr: {corr_out:>8.4f}  (original — outgoing deps)")
    print(f"    W_cooc spatial corr: {corr_cooc:>8.4f}  (co-dependency — NEW)")
    print(f"    W_naive spatial corr:{corr_naive:>8.4f}  (degree-based benchmark)")

    best = max([("W_out", corr_out), ("W_cooc", corr_cooc),
                ("W_naive", corr_naive)], key=lambda x: abs(x[1]))
    print(f"\n    Best W for spatial modelling: {best[0]} (|corr|={abs(best[1]):.4f})")

    # ── Centroid recommendation from W_cooc ───────────────────────────────────

    cooc_connected = has_nb_cooc.sum()
    base = cooc_connected * 0.001

    if abs(corr_cooc) > 0.3:
        mults = [1, 3, 6, 12, 24]
        roughness = "moderate-rough"
    elif abs(corr_cooc) > 0.1:
        mults = [1, 3, 6, 12]
        roughness = "smooth"
    else:
        mults = [1, 3, 6]
        roughness = "very smooth"

    cent_counts = sorted(set([max(3, int(base * m)) for m in mults]))
    cent_counts = [c for c in cent_counts if c <= 500]

    print(f"\n  Centroid recommendation (from W_cooc):")
    print(f"    W_cooc connected nodes: {cooc_connected:>8,}")
    print(f"    Spatial roughness:      {roughness}")
    print(f"    Recommended range:      {cent_counts}")

    # ── Save W_cooc for use in choice model ───────────────────────────────────
    sparse.save_npz(
        os.path.join(DATA_DIR, f"W_cooc_{plat}.npz"), W_cooc)
    sparse.save_npz(
        os.path.join(DATA_DIR, f"W_naive_{plat}.npz"), W_naive)
    print(f"\n    Saved: W_cooc_{plat}.npz  |  W_naive_{plat}.npz")

    all_diag.append({
        "platform":            plat,
        "n_nodes":             sz,
        "n_edges":             len(src_v),
        "corr_W_out":          corr_out,
        "corr_W_cooc":         corr_cooc,
        "corr_W_naive":        corr_naive,
        "gini_W_cooc":         gini_cooc,
        "cooc_connected":      int(cooc_connected),
        "recommended_centroids": str(cent_counts),
    })

# ── Final summary ─────────────────────────────────────────────────────────────

section("Final summary")

diag_df = pd.DataFrame(all_diag)
print(f"\n{'Platform':<12} {'W_out corr':>11} {'W_cooc corr':>12} "
      f"{'W_naive corr':>13} {'Rec. centroids'}")
print("-" * 70)
for _, row in diag_df.iterrows():
    print(f"{row['platform']:<12} {row['corr_W_out']:>11.4f} "
          f"{row['corr_W_cooc']:>12.4f} "
          f"{row['corr_W_naive']:>13.4f}  "
          f"{row['recommended_centroids']}")

diag_df.to_csv(os.path.join(DATA_DIR, "w_diagnostics.csv"), index=False)
print(f"\nSaved: w_diagnostics.csv")
print("W matrices saved as .npz files — ready for 05_choicespatial.py")
