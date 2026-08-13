import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

DATA_DIR = "data/prepared"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

PLATFORMS = ["NPM", "Pypi", "Packagist"]
COLORS    = {"NPM": "#4C9BE8", "Pypi": "#E8834C", "Packagist": "#4CE8A0"}
SEED      = 42

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  ✓ saved: {name}")


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
nodes   = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_utility.parquet"))
eco     = pd.read_csv(os.path.join(DATA_DIR, "ecosystem_metrics.csv"))
results = pd.read_csv(os.path.join(DATA_DIR, "choice_model_results.csv"))
print(f"  Nodes: {len(nodes):,}  |  Eco metrics: {len(eco)} platforms")
print(f"  Choice model results: {len(results)} rows")
print(f"  Models: {results['model'].unique()}")
print(f"  Platforms: {results['platform'].unique()}")


# ── Figure 1: Ecosystem metrics ───────────────────────────────────────────────

print("\nFigure 1: Ecosystem metrics comparison...")

metrics = [
    ("gini_indegree", "Gini Coefficient\n(In-degree Concentration)",
     "Higher = more concentrated\ndependency on few keystones"),
    ("modularity",    "Network Modularity",
     "z-score vs null is\ncomparable (see appendix)"),
    ("pct_in_core",   "% Packages in K-Core",
     "Higher = larger governance-\ncritical core"),
    ("giant_wcc_pct", "Giant Component Size (%)",
     "Higher = more integrated\necosystem"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    "Figure 1: Dependency Architecture by Platform\nCross-platform governance comparison",
    fontsize=12, y=1.02)

for ax, (col, label, note) in zip(axes, metrics):
    vals = [eco.loc[eco["platform"] == p, col].values[0] for p in PLATFORMS]
    bars = ax.bar(PLATFORMS, vals, color=[COLORS[p] for p in PLATFORMS],
                  edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_title(label, fontsize=11, pad=8)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        # Raw modularity is not comparable across sizes: annotate the number of
        # communities (and the null-model z-score where available) so the
        # size dependence is explicit; the standardised z-score is the
        # comparable quantity and is tabulated in the appendix.
        if col == "modularity":
            p = PLATFORMS[i]
            row = eco.loc[eco["platform"] == p]
            extra = ""
            if "n_communities" in eco.columns:
                extra += f"c={int(row['n_communities'].values[0]):,}"
            if "modularity_z" in eco.columns and pd.notna(row['modularity_z'].values[0]):
                extra += f"\nz={row['modularity_z'].values[0]:.1f}"
            if extra:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 0.5, extra,
                        ha="center", va="center", fontsize=6.5, color="black")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_xlabel(note, fontsize=8, color="grey")

plt.tight_layout()
save("01_ecosystem_metrics.png")


# ── Figure 2: Degree distribution ────────────────────────────────────────────

print("Figure 2: Degree distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Figure 2: In-Degree Distribution (Log-Log Scale)\n"
    "Power-law concentration of dependency — keystone dominance",
    fontsize=12, y=1.02)

for ax, plat in zip(axes, PLATFORMS):
    sub     = nodes[nodes["platform"] == plat]["in_degree"]
    nonzero = sub[sub > 0]
    if len(nonzero) > 50_000:
        nonzero = nonzero.sample(50_000, random_state=SEED)
    counts = nonzero.value_counts().sort_index()
    ax.scatter(counts.index, counts.values,
               color=COLORS[plat], alpha=0.5, s=8, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(plat, fontsize=12)
    ax.set_xlabel("In-degree (log scale)")
    ax.set_ylabel("Frequency (log scale)" if plat == "NPM" else "")

    top3 = nodes[(nodes["platform"] == plat) & (nodes["in_degree"] > 0)]\
               .nlargest(3, "in_degree")[["name", "in_degree"]]
    for i, (_, row) in enumerate(top3.iterrows()):
        cnt = counts.get(row["in_degree"], 1)
        ax.annotate(row["name"],
                    xy=(row["in_degree"], cnt),
                    xytext=(row["in_degree"] * 0.15, cnt * (2 + i * 2)),
                    fontsize=7, color="black",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

plt.tight_layout()
save("02_degree_distribution.png")


# ── Figure 3: Utility distribution ───────────────────────────────────────────

print("Figure 3: Utility distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Figure 3: Complementor Utility Distribution by Platform\n"
    "Governance regime shapes utility concentration",
    fontsize=12, y=1.02)

for ax, plat in zip(axes, PLATFORMS):
    sub = nodes[(nodes["platform"] == plat) &
                (nodes["utility_index"].notna()) &
                (nodes["utility_index"] > 0)]["utility_index"]
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=SEED)
    sns.histplot(sub, ax=ax, color=COLORS[plat], bins=60,
                 edgecolor="white", linewidth=0.3, stat="density")
    ax.set_title(plat, fontsize=12)
    ax.set_xlabel("Utility Index (packages with utility > 0)")
    ax.set_ylabel("Density" if plat == "NPM" else "")
    ax.axvline(sub.mean(),   color="black", lw=1.2, ls="--",
               label=f"Mean={sub.mean():.3f}")
    ax.axvline(sub.median(), color="grey",  lw=1.2, ls=":",
               label=f"Median={sub.median():.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
save("03_utility_distribution.png")


# ── Figure 4: Network position vs utility ─────────────────────────────────────

print("Figure 4: Network position vs utility...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Figure 4: Network Position vs. Complementor Utility\n"
    "Structural embeddedness predicts utility — M2 finding",
    fontsize=12, y=1.02)

for ax, plat in zip(axes, PLATFORMS):
    sub = nodes[(nodes["platform"] == plat) &
                (nodes["utility_index"].notna()) &
                (nodes["kcore"] > 0)]
    sub = sub.sample(min(5_000, len(sub)), random_state=SEED)
    sc  = ax.scatter(sub["kcore"], sub["utility_index"],
                     c=np.log(sub["in_degree"] + 1),
                     cmap="YlOrRd", alpha=0.4, s=10, linewidths=0)
    ax.set_title(plat, fontsize=12)
    ax.set_xlabel("K-Core (structural embeddedness)")
    ax.set_ylabel("Utility Index" if plat == "NPM" else "")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log(In-degree)", fontsize=8)

plt.tight_layout()
save("04_position_vs_utility.png")


# ── Figure 5: Keystone packages ───────────────────────────────────────────────

print("Figure 5: Keystone packages...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Figure 5: Top 15 Keystone Packages by PageRank\n"
    "Boundary resource identification — 'too central to fail'",
    fontsize=12, y=1.02)

for ax, plat in zip(axes, PLATFORMS):
    top = nodes[nodes["platform"] == plat]\
              .nlargest(15, "pagerank")[["name", "pagerank", "in_degree", "kcore"]]\
              .sort_values("pagerank")
    bars = ax.barh(top["name"], top["pagerank"],
                   color=COLORS[plat], edgecolor="white", linewidth=0.5)
    ax.set_title(plat, fontsize=12)
    ax.set_xlabel("PageRank")
    ax.tick_params(axis="y", labelsize=8)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"  k={int(row['kcore'])}  dep={int(row['in_degree']):,}",
                va="center", fontsize=7, color="grey")
    ax.set_xlim(0, top["pagerank"].max() * 1.5)

plt.tight_layout()
save("05_keystone_packages.png")


# ── Figure 6: Adj. R² progression M1 → M2 → M3 → M4 ─────────────────────────

print("Figure 6: Adj. R² progression...")

MODEL_COLORS = {
    "M1_quality":   "#B0BEC5",
    "M2_position":  "#5C6BC0",
    "M3_community": "#FFA726",
    "M4_spillover": "#26A69A",
}
MODEL_LABELS = {
    "M1_quality":   "M1: Quality only",
    "M2_position":  "M2: + Governance position",
    "M3_community": "M3: + Community signals",
    "M4_spillover": "M4: + Neighbourhood spillover",
}
MODELS_ORDERED = ["M1_quality", "M2_position", "M3_community", "M4_spillover"]

adj_r2 = results.drop_duplicates(
    subset=["platform", "model"])[["platform", "model", "adj_r2"]].copy()

adj_r2_pivot = adj_r2.pivot_table(
    index="platform", columns="model", values="adj_r2"
).reindex(PLATFORMS)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    "Figure 6: Adj. R² Progression M1 → M2 → M3 → M4 by Platform\n"
    "Incremental explanatory power of quality, position, community, and neighbourhood",
    fontsize=12)

x     = np.arange(len(PLATFORMS))
width = 0.2

for i, model in enumerate(MODELS_ORDERED):
    vals   = [adj_r2_pivot.loc[p, model]
              if p in adj_r2_pivot.index and model in adj_r2_pivot.columns
              else 0 for p in PLATFORMS]
    offset = (i - 1.5) * width
    bars   = ax.bar(x + offset, vals, width=width,
                    color=MODEL_COLORS[model],
                    edgecolor="white", linewidth=0.8,
                    label=MODEL_LABELS[model])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(PLATFORMS, fontsize=11)
ax.set_ylabel("Adjusted R²")
ax.set_ylim(0, 0.65)
ax.legend(fontsize=9, frameon=False, loc="upper left")

plt.tight_layout()
save("06_adj_r2_progression.png")


# ── Figure 7: β₂/β₁ ratio ────────────────────────────────────────────────────

print("Figure 7: β₂/β₁ ratio comparison...")

ratio_data = results.drop_duplicates(
    subset=["platform", "model"]
)[["platform", "model", "b2_b1_ratio"]].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "Figure 7: β₂/β₁ Ratio — Governance Position vs Quality\n"
    "Higher ratio = structural position dominates quality in adoption decisions",
    fontsize=12)

vals   = [ratio_data.loc[
              (ratio_data["platform"] == p) &
              (ratio_data["model"] == "M2_position"),
              "b2_b1_ratio"
          ].values[0] if len(ratio_data.loc[
              (ratio_data["platform"] == p) &
              (ratio_data["model"] == "M2_position")
          ]) > 0 else 0 for p in PLATFORMS]

bars = ax.bar(PLATFORMS, vals,
              color=[COLORS[p] for p in PLATFORMS],
              edgecolor="white", linewidth=0.8, width=0.5)

ax.axhline(1, color="black", lw=1, ls="--", alpha=0.5)
ax.text(len(PLATFORMS) - 0.6, 1.08,
        "β₂/β₁ = 1 (balanced)", fontsize=8, color="grey")

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{v:.2f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

ax.set_ylabel("β₂/β₁ Ratio")
ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
save("07_beta_ratio_comparison.png")


# ── Figure 8: Neighbourhood spillover β₄ ─────────────────────────────────────

print("Figure 8: Neighbourhood spillover β₄...")

spillover = results[
    (results["model"] == "M4_spillover") &
    (results["variable"] == "x4_neighbourhood_z")
][["platform", "coef", "se"]].copy()

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "Figure 8: Neighbourhood Spillover Effect (β₄) by Platform\n"
    "Positive = additive adoption culture  |  Negative = competitive displacement",
    fontsize=12)

colors_list = [COLORS[p] for p in spillover["platform"]]
bars = ax.bar(spillover["platform"], spillover["coef"],
              yerr=spillover["se"],
              color=colors_list, edgecolor="white", linewidth=0.8,
              capsize=5, error_kw={"elinewidth": 1.5})

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("β₄ coefficient (standardised)")

max_v = spillover["coef"].abs().max()
ax.set_ylim(-max_v * 0.5, max_v * 1.5)

for bar, (_, row) in zip(bars, spillover.iterrows()):
    c    = row["coef"]
    ypos = c + max_v * 0.05 if c >= 0 else c - max_v * 0.05
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{c:.4f}***", ha="center",
            va="bottom" if c >= 0 else "top",
            fontsize=10, fontweight="bold")

plt.tight_layout()
save("08_spillover_comparison.png")


# ── Figure 9: Coefficient heatmap ─────────────────────────────────────────────

print("Figure 9: Coefficient heatmap (M2 position variables)...")

M2_VARS = ["x2_kcore_z", "x2_clustering_z", "x2_betweenness_z"]
VAR_LABELS = {
    "x2_kcore_z":       "K-Core",
    "x2_clustering_z":  "Clustering",
    "x2_betweenness_z": "Betweenness",
}

m2_coefs = results[
    (results["model"] == "M2_position") &
    (results["variable"].isin(M2_VARS))
][["platform", "variable", "coef"]].copy()

m2_pivot = m2_coefs.pivot_table(
    index="variable", columns="platform", values="coef"
).reindex(M2_VARS)
m2_pivot.index = [VAR_LABELS[v] for v in m2_pivot.index]

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle(
    "Figure 9: Governance Position Coefficients by Platform (M2)\n"
    "Standardised coefficients — cross-platform comparison",
    fontsize=12)

sns.heatmap(m2_pivot, ax=ax, annot=True, fmt=".4f",
            cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Standardised coefficient"})

ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
save("09_coefficient_heatmap.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}/")
print("\nFigure summary:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"  {f:<45} {size:>6.0f} KB")
