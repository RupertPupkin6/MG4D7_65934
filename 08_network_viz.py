import os
import warnings
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

DATA_DIR = "data/prepared"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

PLATFORMS = ["NPM", "Pypi", "Packagist"]
COLORS    = {"NPM": "#4C9BE8", "Pypi": "#E8834C", "Packagist": "#4CE8A0"}

KEYSTONES = {
    "NPM":       "lodash",
    "Pypi":      "requests",
    "Packagist": "guzzlehttp/guzzle",
}

MAX_KCORE_NODES = 300
MAX_HOP1_SHOW   = 80
MAX_EGO_HOP2    = 40
SEED            = 42

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    10,
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
})

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  ✓ saved: {name}")


# ── Layout helper ─────────────────────────────────────────────────────────────

def fr_layout(g):
    np.random.seed(SEED)
    n = g.vcount()
    if n == 0:
        return np.array([]), np.array([])
    init   = np.random.uniform(0, 1, (n, 2)).tolist()
    layout = g.layout_fruchterman_reingold(seed=init)
    coords = np.array(layout.coords)
    return coords[:, 0], coords[:, 1]


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_utility.parquet"))
edges = pd.read_parquet(os.path.join(DATA_DIR, "edges.parquet"))
print(f"  Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")


# ── Build igraph per platform ─────────────────────────────────────────────────

def build_igraph(platform):
    n   = nodes[nodes["platform"] == platform].reset_index(drop=True)
    e   = edges[edges["platform"] == platform]
    idx = {name: i for i, name in enumerate(n["name"])}

    src   = e["project_name"].map(idx).dropna().astype(int)
    tgt   = e["dependency_name"].map(idx).dropna().astype(int)
    valid = src.index.intersection(tgt.index)
    src_v = src.loc[valid].values
    tgt_v = tgt.loc[valid].values

    g = ig.Graph(n=len(n), edges=list(zip(src_v, tgt_v)), directed=True)
    g.vs["name"]      = n["name"].tolist()
    g.vs["pagerank"]  = n["pagerank"].fillna(0).tolist()
    g.vs["in_degree"] = n["in_degree"].fillna(0).astype(int).tolist()
    g.vs["utility"]   = n["utility_index"].fillna(0).tolist()
    g.vs["kcore"]     = ig.Graph.as_undirected(g, mode="collapse").coreness()
    return g, idx


# ── Figure 10: K-core subgraph ────────────────────────────────────────────────

print("\nFigure 10: K-core subgraph...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "Figure 10: Governance-Critical Core — K-Core Subgraph by Platform\n"
    "Nodes sized by PageRank  ·  Colour = k-core shell depth  ·  "
    "Edges = runtime dependencies",
    fontsize=11, y=1.03
)

cmap = plt.colormaps["YlOrRd"]   # replaces deprecated cm.get_cmap()

for ax, plat in zip(axes, PLATFORMS):
    print(f"  [{plat}] building k-core subgraph...")
    g, _ = build_igraph(plat)

    cores   = g.vs["kcore"]
    max_k   = max(cores)
    shell_k = max(2, max_k - 2)
    keep    = [v.index for v in g.vs if v["kcore"] >= shell_k]

    print(f"    max k={max_k}, plotting shell >= {shell_k}: {len(keep)} nodes")

    if len(keep) > MAX_KCORE_NODES:
        keep = sorted(keep,
                      key=lambda i: g.vs[i]["pagerank"],
                      reverse=True)[:MAX_KCORE_NODES]

    sg   = g.induced_subgraph(keep)
    x, y = fr_layout(sg)

    if len(x) == 0:
        ax.set_title(f"{plat} — no nodes")
        ax.axis("off")
        continue

    pr   = np.array(sg.vs["pagerank"])
    pr_n = (pr - pr.min()) / (pr.max() - pr.min() + 1e-10)
    sizes = 20 + pr_n * 300

    kc     = np.array(sg.vs["kcore"])
    norm   = mcolors.Normalize(vmin=kc.min(), vmax=kc.max())
    colors = cmap(norm(kc))

    for e_obj in sg.es:
        s, t = e_obj.source, e_obj.target
        ax.plot([x[s], x[t]], [y[s], y[t]],
                color="grey", alpha=0.15, lw=0.4, zorder=1)

    ax.scatter(x, y, s=sizes, c=colors, zorder=2,
               edgecolors="white", linewidths=0.3, alpha=0.9)

    for i in np.argsort(pr)[-8:]:
        ax.annotate(sg.vs[i]["name"],
                    xy=(x[i], y[i]),
                    fontsize=6.5,
                    fontweight="bold" if pr_n[i] > 0.7 else "normal",
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", alpha=0.7, lw=0))

    ax.set_title(
        f"{plat}  —  k-core ≥ {shell_k}  ({len(keep)} nodes shown)",
        fontsize=11)
    ax.axis("off")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("K-core shell", fontsize=8)

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
           markersize=s, label=l)
    for s, l in [(4, "Low PageRank"), (8, "Med PageRank"), (14, "High PageRank")]
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, frameon=False,
           title="Node size = PageRank", title_fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
save("10_kcore_subgraph.png")


# ── Figure 11: Ego networks ───────────────────────────────────────────────────

print("\nFigure 11: Ego networks...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    "Figure 11: Ego Networks of Top Keystone Packages\n"
    "1-hop neighbours shown in full  ·  2-hop sample in grey\n"
    "Boundary resource structural position",
    fontsize=11, y=1.03
)

for ax, plat in zip(axes, PLATFORMS):
    keystone = KEYSTONES[plat]
    print(f"  [{plat}] ego network of '{keystone}'...")

    g, idx = build_igraph(plat)

    if keystone not in idx:
        ax.set_title(f"{plat}\n'{keystone}' not found")
        ax.axis("off")
        continue

    ego_idx = idx[keystone]

    hop1_out = set(g.neighbors(ego_idx, mode="out"))
    hop1_in  = set(g.neighbors(ego_idx, mode="in"))

    hop1_in_sorted = sorted(
        list(hop1_in),
        key=lambda i: g.vs[i]["in_degree"],
        reverse=True
    )
    hop1_in_show  = set(hop1_in_sorted[:MAX_HOP1_SHOW])
    total_hop1_in = len(hop1_in)

    hop1_out_show = hop1_out
    hop1_show     = hop1_in_show | hop1_out_show

    hop2 = set()
    for nb in hop1_show:
        hop2.update(g.neighbors(nb, mode="all"))
    hop2 = hop2 - hop1_show - {ego_idx}

    np.random.seed(SEED)
    if len(hop2) > MAX_EGO_HOP2:
        hop2 = set(np.random.choice(list(hop2), MAX_EGO_HOP2, replace=False))

    all_nodes   = [ego_idx] + list(hop1_show) + list(hop2)
    sg          = g.induced_subgraph(all_nodes)
    old_to_new  = {old: new for new, old in enumerate(all_nodes)}
    ego_new     = old_to_new[ego_idx]
    hop1_in_new = {old_to_new[i] for i in hop1_in_show  if i in old_to_new}
    hop1_out_new= {old_to_new[i] for i in hop1_out_show if i in old_to_new}
    hop2_new    = {old_to_new[i] for i in hop2          if i in old_to_new}

    x, y = fr_layout(sg)

    if len(x) == 0:
        ax.set_title(f"{plat} — no nodes")
        ax.axis("off")
        continue

    for e_obj in sg.es:
        s, t   = e_obj.source, e_obj.target
        is_ego = (s == ego_new or t == ego_new)
        ax.plot([x[s], x[t]], [y[s], y[t]],
                color=COLORS[plat] if is_ego else "lightgrey",
                alpha=0.6 if is_ego else 0.3,
                lw=0.8 if is_ego else 0.4, zorder=1)

    h2 = list(hop2_new)
    if h2:
        ax.scatter(x[h2], y[h2], s=15, color="lightgrey",
                   edgecolors="white", linewidths=0.2, zorder=2, alpha=0.7)

    h1_in = list(hop1_in_new)
    if h1_in:
        ax.scatter(x[h1_in], y[h1_in], s=40, color=COLORS[plat],
                   edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)

    h1_out = list(hop1_out_new)
    if h1_out:
        ax.scatter(x[h1_out], y[h1_out], s=40, color="#E8834C",
                   edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)

    ax.scatter([x[ego_new]], [y[ego_new]], s=300, color="black",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.annotate(keystone, xy=(x[ego_new], y[ego_new]),
                xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="black",
                          ec="none", alpha=0.8),
                color="white")

    top_dep = sorted(list(hop1_in),
                     key=lambda i: g.vs[i]["in_degree"],
                     reverse=True)[:5]
    for i in top_dep:
        ni = old_to_new.get(i)
        if ni is None:
            continue
        ax.annotate(g.vs[i]["name"],
                    xy=(x[ni], y[ni]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", alpha=0.6, lw=0))

    ax.set_title(
        f"{plat}  —  '{keystone}'\n"
        f"Top {len(hop1_in_show)} of {total_hop1_in:,} dependents shown  ·  "
        f"{len(hop2)} 2-hop (sample)",
        fontsize=10)
    ax.axis("off")

    handles = []
    if h1_in:
        handles.append(
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=COLORS[plat], markersize=8,
                   label=f"Top dependents by in-degree "
                         f"({len(h1_in)} shown, {total_hop1_in:,} total)")
        )
    if h2:
        handles.append(
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='lightgrey', markersize=6,
                   label="2-hop neighbourhood (sample)")
        )
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=False)

shared_legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
           markersize=12, label='Keystone package'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#4C9BE8',
           markersize=8,  label='Top dependents (by in-degree)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='lightgrey',
           markersize=6,  label='2-hop neighbourhood (sample)'),
]
fig.legend(handles=shared_legend, loc="lower center", ncol=4,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
save("11_ego_networks.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nAll network figures saved to: {os.path.abspath(FIG_DIR)}/")
print("\nFigure summary:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"  {f:<45} {size:>6.0f} KB")
