import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse
from scipy.stats import norm, chi2

warnings.filterwarnings("ignore")

DATA_DIR  = "data/prepared"
PLATFORMS = ["NPM", "Pypi", "Packagist"]

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Load data ─────────────────────────────────────────────────────────────────

section("Loading data")
nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_utility.parquet"))
print(f"Nodes: {len(nodes):,}")


# ── Prepare node attributes ───────────────────────────────────────────────────

section("Preparing node attributes")

nodes = nodes.copy()
nodes["age_days"]           = nodes["age_days"].fillna(0).clip(lower=0)
nodes["versions_count"]     = nodes["versions_count"].fillna(0)
nodes["has_stable_version"] = nodes["has_stable_version"].fillna(0).astype(int)
nodes["stars_count"]        = nodes["stars_count"].fillna(0)
nodes["contributors_count"] = nodes["contributors_count"].fillna(0)

# x1: intrinsic quality
nodes["x1_log_age"]      = np.log(nodes["age_days"] + 1)
nodes["x1_log_versions"] = np.log(nodes["versions_count"] + 1)
nodes["x1_stable"]       = nodes["has_stable_version"]

# x2: governance position
nodes["x2_kcore"]       = nodes["kcore"].fillna(0)
nodes["x2_clustering"]  = nodes["clustering"].fillna(0)
nodes["x2_betweenness"] = nodes["betweenness"].fillna(0)

# x3: community signals
nodes["x3_log_stars"]        = np.log(nodes["stars_count"] + 1)
nodes["x3_log_contributors"] = np.log(nodes["contributors_count"] + 1)

# Dependent variable
nodes["y"] = np.log(nodes["dependent_projects_count"] + 1)

X1_COLS = ["x1_log_age", "x1_log_versions", "x1_stable"]
X2_COLS = ["x2_kcore", "x2_clustering", "x2_betweenness"]
X3_COLS = ["x3_log_stars", "x3_log_contributors"]

print(f"  x1 (quality):    {X1_COLS}")
print(f"  x2 (position):   {X2_COLS}")
print(f"  x3 (community):  {X3_COLS}")
print(f"  y  (adoption):   log(dependent_projects_count + 1)")

for plat in PLATFORMS:
    sub = nodes[nodes["platform"] == plat]
    print(f"  {plat:<15} {len(sub):>10,} packages")


# ── Compute neighbourhood spillover (x4) ─────────────────────────────────────

section("Computing neighbourhood spillover (x4) from W_cooc")

for plat in PLATFORMS:
    w_path = os.path.join(DATA_DIR, f"W_cooc_{plat}.npz")
    if not os.path.exists(w_path):
        print(f"  [{plat}] W_cooc not found — run 05_diagnose_w.py first")
        nodes.loc[nodes["platform"] == plat, "x4_neighbourhood"] = 0.0
        continue

    W      = sparse.load_npz(w_path)
    n_plat = nodes[nodes["platform"] == plat].reset_index(drop=True)
    y      = n_plat["y"].fillna(0).values
    nb     = np.array(W @ y).flatten()

    idx_map = {name: i for i, name in enumerate(n_plat["name"])}
    nodes.loc[nodes["platform"] == plat, "x4_neighbourhood"] = \
        nodes.loc[nodes["platform"] == plat, "name"].map(
            lambda nm: nb[idx_map[nm]] if nm in idx_map else 0.0
        ).values

    n_nonzero = (nb > 0).sum()
    print(f"  [{plat}] {n_nonzero:,} packages with non-zero neighbourhood "
          f"({n_nonzero/len(n_plat)*100:.1f}%)  "
          f"mean={nb[nb>0].mean():.4f}  max={nb.max():.4f}")

nodes["x4_neighbourhood"] = nodes["x4_neighbourhood"].fillna(0)


# ── Model helpers ─────────────────────────────────────────────────────────────

def beta_ratio(result, cols_a, cols_b):
    """L2 norm ratio of two coefficient vectors."""
    b_a = np.array([result.params.get(c, 0) for c in cols_a])
    b_b = np.array([result.params.get(c, 0) for c in cols_b])
    n_a = np.linalg.norm(b_a)
    n_b = np.linalg.norm(b_b)
    return n_b / n_a if n_a > 1e-10 else np.nan


def print_model(name, result, ratios=None):
    print(f"\n  ── {name} ──")
    print(f"     N={int(result.nobs):,}  "
          f"R²={result.rsquared:.4f}  "
          f"Adj.R²={result.rsquared_adj:.4f}  "
          f"AIC={result.aic:.1f}")
    if ratios:
        for label, val in ratios.items():
            if isinstance(val, float):
                print(f"     {label}: {val:.4f}")
    print(f"     {'Variable':<30} {'Coef':>9} {'SE':>9} {'p':>9}")
    print(f"     {'-'*60}")
    for var in result.params.index:
        c   = result.params[var]
        s   = result.bse[var]
        p   = result.pvalues[var]
        sig = "***" if p<0.001 else "** " if p<0.01 \
              else "*  " if p<0.05 else "   "
        print(f"     {var:<30} {c:>9.4f} {s:>9.4f} {p:>9.4f} {sig}")


def collect(name, plat, result, ratios):
    rows = []
    for var in result.params.index:
        rows.append({
            "platform":  plat,
            "model":     name,
            "variable":  var,
            "coef":      result.params[var],
            "se":        result.bse[var],
            "pvalue":    result.pvalues[var],
            "r2":        result.rsquared,
            "adj_r2":    result.rsquared_adj,
            "n":         int(result.nobs),
            **(ratios or {})
        })
    return rows


# ── Robust inference and spatial diagnostics ─────────────────────────────────
#
# OLS assumes independent, identically distributed errors. That assumption is
# violated here: packages are nodes in a dependency network, so neighbouring
# observations are correlated and the classical OLS standard errors understate
# uncertainty. We address this in two ways:
#
#   (1) Robustness — standard errors are clustered on Louvain community
#       (packages in the same community are allowed arbitrary within-group
#       correlation). If community labels are unavailable we fall back to
#       heteroskedasticity-robust HC3 errors. Point estimates and R² are
#       unchanged; only the standard errors and p-values are corrected.
#
#   (2) Score checks — the non-spatial model (M3) residuals are formally tested
#       for spatial autocorrelation with Moran's I and the Anselin (1988)
#       Lagrange-Multiplier (score) tests LM-lag and LM-error, using the same
#       co-dependency weight matrix W_cooc that defines the spillover term. If
#       these reject independence, the M4 spatial-lag term is warranted; the
#       Moran's I of M4 residuals is reported to show the remaining dependence.
#
# Full spatial-autoregressive estimation (ML / GMM, e.g. PySAL spreg) is left to
# further research; at this sample size the diagnostics + clustered errors +
# spillover term give an honest, tractable treatment of the dependence.

def robust_cov_args(df):
    """Cluster-robust on community when available, else HC3."""
    if "community" in df.columns and df["community"].notna().any():
        return {"cov_type": "cluster",
                "cov_kwds": {"groups": df["community"].astype(int).values}}
    return {"cov_type": "HC3"}


def morans_i(e, W):
    """Moran's I of vector e under weight matrix W, with normality-based z, p."""
    n = len(e)
    W = W.tocsr()
    S0 = W.sum()
    if S0 == 0:
        return np.nan, np.nan, np.nan, np.nan
    e = e - e.mean()
    I  = (n / S0) * ((e @ (W @ e)) / (e @ e))
    EI = -1.0 / (n - 1)
    WpWt = W + W.transpose().tocsr()
    S1   = 0.5 * WpWt.multiply(WpWt).sum()
    rs   = np.asarray(W.sum(1)).ravel()
    cs   = np.asarray(W.sum(0)).ravel()
    S2   = np.sum((rs + cs) ** 2)
    n2   = n * n
    varI = (n2*S1 - n*S2 + 3*S0*S0) / ((n2 - 1)*S0*S0) - EI*EI
    z = (I - EI) / np.sqrt(varI) if varI > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return I, EI, z, p


def lm_tests(e, y, X, fitted, W):
    """Anselin LM-error and LM-lag score tests on OLS residuals (each chi2(1))."""
    n  = len(e)
    W  = W.tocsr()
    s2 = (e @ e) / n
    if s2 == 0:
        return (np.nan,)*4
    T1 = W.multiply(W).sum() + W.multiply(W.transpose()).sum()   # tr[(W'+W)W]
    We = W @ e
    Wy = W @ y
    Wf = W @ fitted                                              # W X beta_hat
    lm_err = ((e @ We) / s2) ** 2 / T1
    XtX_inv = np.linalg.inv(X.T @ X)
    XtWf    = X.T @ Wf
    WfMWf   = (Wf @ Wf) - XtWf @ (XtX_inv @ XtWf)               # (WXb)'M(WXb)
    D = WfMWf / s2 + T1
    lm_lag = ((e @ Wy) / s2) ** 2 / D
    return (lm_err, 1 - chi2.cdf(lm_err, 1),
            lm_lag, 1 - chi2.cdf(lm_lag, 1))


# ── Package-level models ──────────────────────────────────────────────────────

section("PACKAGE-LEVEL MODELS (one row per package)")

all_results = []
spatial_diagnostics = []

for plat in PLATFORMS:
    print(f"\n{'─'*60}")
    print(f"  Platform: {plat}")
    print(f"{'─'*60}")

    df = nodes[nodes["platform"] == plat].copy()
    df = df[df["versions_count"] > 0].copy()

    # Standardise within platform
    for col in X1_COLS + X2_COLS + X3_COLS + ["x4_neighbourhood"]:
        mu  = df[col].mean()
        std = df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / std if std > 0 else 0.0

    z1 = [f"{c}_z" for c in X1_COLS]
    z2 = [f"{c}_z" for c in X2_COLS]
    z3 = [f"{c}_z" for c in X3_COLS]
    z4 = ["x4_neighbourhood_z"]

    cov = robust_cov_args(df)
    print(f"  Robust SEs: {cov['cov_type']}"
          + (f" on '{'community'}'" if cov['cov_type'] == 'cluster' else ""))

    # M1 — Quality only
    m1 = smf.ols("y ~ " + " + ".join(z1), data=df).fit(**cov)
    print_model("M1 — Quality only", m1)
    all_results += collect("M1_quality", plat, m1, None)

    # M2 — Quality + Position
    m2   = smf.ols("y ~ " + " + ".join(z1 + z2), data=df).fit(**cov)
    r2   = beta_ratio(m2, z1, z2)
    lft2 = m2.rsquared_adj - m1.rsquared_adj
    print_model("M2 — Quality + Position", m2,
                ratios={"β₂/β₁": r2, "Adj.R² lift over M1": lft2})
    all_results += collect("M2_position", plat, m2,
                           {"b2_b1_ratio": r2, "adj_r2_lift": lft2})

    # M3 — Quality + Position + Community
    m3   = smf.ols("y ~ " + " + ".join(z1 + z2 + z3), data=df).fit(**cov)
    r3   = beta_ratio(m3, z1, z3)
    lft3 = m3.rsquared_adj - m2.rsquared_adj
    print_model("M3 — Quality + Position + Community", m3,
                ratios={"β₃/β₁": r3, "Adj.R² lift over M2": lft3})
    all_results += collect("M3_community", plat, m3,
                           {"b3_b1_ratio": r3, "adj_r2_lift": lft3})

    # M4 — Quality + Position + Community + Neighbourhood spillover
    m4   = smf.ols("y ~ " + " + ".join(z1 + z2 + z3 + z4), data=df).fit(**cov)
    b4   = m4.params.get("x4_neighbourhood_z", np.nan)
    p4   = m4.pvalues.get("x4_neighbourhood_z", np.nan)
    lft4 = m4.rsquared_adj - m3.rsquared_adj
    sig4 = "***" if p4<0.001 else "**" if p4<0.01 \
           else "*" if p4<0.05 else "n.s."
    print_model("M4 — + Neighbourhood spillover", m4,
                ratios={"β₄ spillover": b4,
                        "p(β₄)": p4,
                        "Adj.R² lift over M3": lft4})
    print(f"\n     Spillover: β₄={b4:.4f}  p={p4:.4f}  {sig4}")
    all_results += collect("M4_spillover", plat, m4,
                           {"b4_spillover": b4, "adj_r2_lift": lft4})

    # ── Spatial dependence diagnostics on M3 / M4 residuals ───────────────────
    w_path = os.path.join(DATA_DIR, f"W_cooc_{plat}.npz")
    if os.path.exists(w_path):
        W_full   = sparse.load_npz(w_path)
        n_full   = nodes[nodes["platform"] == plat].reset_index(drop=True)
        full_idx = {nm: i for i, nm in enumerate(n_full["name"])}
        keep     = [full_idx[nm] for nm in df["name"].values if nm in full_idx]

        if len(keep) == len(df):
            # restrict W to the estimation sample
            Ws = W_full[keep][:, keep].tocsr()

            # Further restrict to units with >=1 co-dependency neighbour. With
            # ~70-98% of packages isolated in W_cooc, keeping them makes n/S0
            # explode and inflates Moran's I beyond its valid range (and the LM
            # statistics with it). Isolated units carry no spatial information,
            # so the dependence test is defined on the connected sub-population;
            # n_connected is reported for transparency.
            deg  = np.asarray((Ws > 0).sum(1)).ravel()
            conn = np.where(deg > 0)[0]

            if len(conn) >= 50:
                Wc = Ws[conn][:, conn].tocsr()
                rc = np.asarray(Wc.sum(1)).ravel(); rc[rc == 0] = 1
                Wc = sparse.diags(1.0 / rc) @ Wc

                e3  = m3.resid.values[conn]
                f3  = m3.fittedvalues.values[conn]
                X3  = m3.model.exog[conn]
                y3  = df["y"].values[conn]
                e4  = m4.resid.values[conn]

                I3, EI3, z3m, p3m = morans_i(e3, Wc)
                le, ple, ll, pll  = lm_tests(e3, y3, X3, f3, Wc)
                I4, EI4, z4m, p4m = morans_i(e4, Wc)

                print(f"\n     ── Spatial diagnostics (W_cooc, {len(conn):,} "
                      f"connected units) ──")
                print(f"     Moran's I (M3 resid): {I3:+.4f}  E[I]={EI3:+.5f}  "
                      f"z={z3m:+.2f}  p={p3m:.3g}")
                print(f"     LM-error (M3):        {le:12.2f}  p={ple:.3g}")
                print(f"     LM-lag   (M3):        {ll:12.2f}  p={pll:.3g}")
                print(f"     Moran's I (M4 resid): {I4:+.4f}  z={z4m:+.2f}  "
                      f"p={p4m:.3g}   (spatial lag included)")

                spatial_diagnostics.append({
                    "platform":      plat,
                    "n_estimation":  len(df),
                    "n_connected":   int(len(conn)),
                    "moran_I_M3":    I3, "moran_z_M3": z3m, "moran_p_M3": p3m,
                    "LM_error_M3":   le, "LM_error_p_M3": ple,
                    "LM_lag_M3":     ll, "LM_lag_p_M3":   pll,
                    "moran_I_M4":    I4, "moran_z_M4": z4m, "moran_p_M4": p4m,
                    "cov_type":      cov["cov_type"],
                })
            else:
                print(f"\n     [spatial diagnostics skipped: only {len(conn)} "
                      f"connected units]")
        else:
            print(f"\n     [spatial diagnostics skipped: name alignment mismatch]")
    else:
        print(f"\n     [spatial diagnostics skipped: {w_path} not found]")


# ── Cross-platform summary ────────────────────────────────────────────────────

section("Cross-platform summary")

results_df = pd.DataFrame(all_results)

print("\n  Adj. R² progression by platform:")
tbl = results_df.drop_duplicates(subset=["platform", "model"])[
    ["platform", "model", "adj_r2"]
].pivot_table(index="platform", columns="model", values="adj_r2")
print(tbl.round(4).to_string())

print("\n  β₂/β₁ ratio (governance position vs quality) — M2:")
r2_tbl = results_df[
    (results_df["model"] == "M2_position")
].drop_duplicates(subset=["platform"])[
    ["platform", "b2_b1_ratio"]
]
print(r2_tbl.to_string(index=False))

print("\n  β₃/β₁ ratio (community vs quality) — M3:")
r3_tbl = results_df[
    (results_df["model"] == "M3_community")
].drop_duplicates(subset=["platform"])[
    ["platform", "b3_b1_ratio"]
]
print(r3_tbl.to_string(index=False))

print("\n  Neighbourhood spillover β₄ (M4):")
sp = results_df[
    (results_df["model"] == "M4_spillover") &
    (results_df["variable"] == "x4_neighbourhood_z")
][["platform", "coef", "se", "pvalue"]].copy()
sp["sig"] = sp["pvalue"].apply(
    lambda p: "***" if p<0.001 else "**" if p<0.01
              else "*" if p<0.05 else "n.s.")
print(sp.to_string(index=False))

print("""
  ── Governance interpretation ────────────────────────────────
  β₂/β₁ > 1 → structural position dominates quality
              (architectural self-reinforcement)
  β₂/β₁ < 1 → quality dominates position (healthy governance)

  β₄ > 0    → additive neighbourhood culture
              (co-adoption with high-adoption packages helps)
  β₄ < 0    → competitive displacement
              (co-adoption with high-adoption packages hurts)

  Cross-platform differences in β₂/β₁ and β₄ reveal how
  governance regime shapes adoption dynamics.
  ────────────────────────────────────────────────────────────
""")


# ── Save ──────────────────────────────────────────────────────────────────────

section("Saving outputs")

results_df.to_csv(
    os.path.join(DATA_DIR, "choice_model_results.csv"), index=False)
print(f"Saved: choice_model_results.csv ({len(results_df)} rows)")

if spatial_diagnostics:
    diag_df = pd.DataFrame(spatial_diagnostics)
    print("\n  Spatial dependence diagnostics (for appendix):")
    print("  OLS independence is rejected where Moran's I / LM p-values are small.")
    print(diag_df.round(4).to_string(index=False))
    diag_df.to_csv(
        os.path.join(DATA_DIR, "ols_spatial_diagnostics.csv"), index=False)
    print(f"\nSaved: ols_spatial_diagnostics.csv ({len(diag_df)} rows)")

print("\nChoice modelling complete. Ready for visualisation.")
