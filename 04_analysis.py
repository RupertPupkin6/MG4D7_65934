import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_DIR  = "data/prepared"
PLATFORMS = ["NPM", "Pypi", "Packagist"]

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Load data ─────────────────────────────────────────────────────────────────

section("Loading data")
nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_metrics.parquet"))
print(f"Loaded: {len(nodes):,} rows x {nodes.shape[1]} columns")


# ── Prepare dataset ───────────────────────────────────────────────────────────

section("Preparing dataset")

df = nodes[nodes["versions_count"] > 0].copy()

df["sourcerank"]         = df["sourcerank"].fillna(0)
df["age_days"]           = df["age_days"].fillna(0).clip(lower=0)
df["versions_count"]     = df["versions_count"].fillna(0)
df["has_stable_version"] = df["has_stable_version"].astype(int)

print(f"Dataset: {len(df):,} rows")
for plat in PLATFORMS:
    print(f"  {plat:<15} {(df['platform']==plat).sum():>10,}")


# ── Construct utility index ───────────────────────────────────────────────────

section("Constructing utility index")

scaler     = MinMaxScaler()
components = ["log_dependent_projects", "pagerank", "kcore"]

df["log_dependent_projects"] = np.log(df["dependent_projects_count"] + 1)

utility_parts = []
for plat in PLATFORMS:
    mask = df["platform"] == plat
    sub  = df.loc[mask, components].copy()
    sub_scaled = pd.DataFrame(
        scaler.fit_transform(sub),
        columns=[f"{c}_scaled" for c in components],
        index=sub.index
    )
    utility_parts.append(sub_scaled)

scaled = pd.concat(utility_parts).sort_index()
df     = df.join(scaled)

df["utility_index"] = (
    df["log_dependent_projects_scaled"] +
    df["pagerank_scaled"] +
    df["kcore_scaled"]
) / 3.0

print("Utility index summary:")
for plat in PLATFORMS:
    u = df.loc[df["platform"] == plat, "utility_index"]
    print(f"  {plat:<15} mean={u.mean():.3f}  median={u.median():.3f}  "
          f"max={u.max():.3f}  zeros={(u==0).mean()*100:.1f}%")


# ── Graph metric summary per platform ────────────────────────────────────────

section("Graph metric summary per platform")

eco_rows = []
for plat in PLATFORMS:
    sub = df[df["platform"] == plat]

    print(f"\n  {plat}")
    print(f"  {'Metric':<35} {'Mean':>10} {'Median':>10} {'Max':>10}")
    print(f"  {'-'*65}")

    # betweenness and pagerank_norm are size-normalised in 03_graph.py, so the
    # cross-platform figures below are directly comparable; raw pagerank is
    # retained only for within-platform ranking.
    for col in ["in_degree", "out_degree", "pagerank_norm", "betweenness",
                "clustering", "kcore"]:
        vals = sub[col].dropna()
        print(f"  {col:<35} {vals.mean():>10.4f} "
              f"{vals.median():>10.4f} {vals.max():>10.4f}")

    eco_rows.append({
        "platform":           plat,
        "n_packages":         len(sub),
        "mean_in_degree":     sub["in_degree"].mean(),
        "max_in_degree":      sub["in_degree"].max(),
        "mean_pagerank_norm": sub["pagerank_norm"].mean(),
        "max_pagerank_norm":  sub["pagerank_norm"].max(),
        "mean_betweenness":   sub["betweenness"].mean(),
        "max_betweenness":    sub["betweenness"].max(),
        "mean_kcore":         sub["kcore"].mean(),
        "max_kcore":          sub["kcore"].max(),
        "mean_clustering":    sub["clustering"].mean(),
        "mean_utility":       sub["utility_index"].mean(),
        "pct_zero_utility":   (sub["utility_index"] == 0).mean() * 100,
    })


# ── Top packages by PageRank per platform ─────────────────────────────────────

section("Top 15 packages by PageRank per platform")

for plat in PLATFORMS:
    sub  = df[df["platform"] == plat]
    top15 = sub.nlargest(15, "pagerank")[
        ["name", "pagerank", "in_degree", "kcore", "utility_index"]
    ]
    print(f"\n  {plat}:")
    print(top15.to_string(index=False))


# ── Utility distribution summary ──────────────────────────────────────────────

section("Utility index distribution per platform")

for plat in PLATFORMS:
    u = df.loc[df["platform"] == plat, "utility_index"]
    print(f"\n  {plat}:")
    print(f"    mean:   {u.mean():.4f}")
    print(f"    median: {u.median():.4f}")
    print(f"    top 1%: {u.quantile(0.99):.4f}")
    print(f"    zeros:  {(u==0).mean()*100:.1f}%")


# ── Save outputs ──────────────────────────────────────────────────────────────

section("Saving outputs")

nodes_out   = nodes.copy()
utility_map = df[["platform", "name", "utility_index",
                  "log_dependent_projects_scaled",
                  "pagerank_scaled", "kcore_scaled"]].copy()
nodes_out   = nodes_out.merge(utility_map, on=["platform", "name"], how="left")
nodes_out.to_parquet(
    os.path.join(DATA_DIR, "nodes_with_utility.parquet"), index=False)

eco_df = pd.DataFrame(eco_rows)
eco_df.to_csv(
    os.path.join(DATA_DIR, "ecosystem_summary.csv"), index=False)

print(f"Saved: nodes_with_utility.parquet")
print(f"Saved: ecosystem_summary.csv")
print(f"\nEcosystem summary:")
print(eco_df.to_string(index=False))
print("\nGraph metric analysis complete. Ready for choice modelling.")
