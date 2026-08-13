import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from scipy import sparse
from scipy.stats import norm, chi2, probplot

warnings.filterwarnings("ignore")

DATA_DIR   = "data/prepared"
FIG_DIR    = "figures"
PLATFORMS  = ["NPM", "Pypi", "Packagist"]

# Config
COMPUTE_WHITE = True        # White test is O(k^2) in regressors; subsampled below
WHITE_MAX_N   = 100_000     # subsample cap for the White auxiliary regression
SAVE_PLOTS    = True        # residual-vs-fitted and QQ plots for M3
PLOT_SAMPLE   = 5_000
SEED          = 42


def section(title):
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


# ── Spatial helpers (same maths as 06_choice_modelling.py) ────────────────────

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
        return (np.nan,) * 4
    T1 = W.multiply(W).sum() + W.multiply(W.transpose()).sum()
    We = W @ e
    Wy = W @ y
    Wf = W @ fitted
    lm_err = ((e @ We) / s2) ** 2 / T1
    XtX_inv = np.linalg.inv(X.T @ X)
    XtWf    = X.T @ Wf
    WfMWf   = (Wf @ Wf) - XtWf @ (XtX_inv @ XtWf)
    D = WfMWf / s2 + T1
    lm_lag = ((e @ Wy) / s2) ** 2 / D
    return (lm_err, 1 - chi2.cdf(lm_err, 1),
            lm_lag, 1 - chi2.cdf(lm_lag, 1))


# ── Load & prepare (mirrors 06_choice_modelling.py exactly) ───────────────────

section("Loading data")
nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes_with_utility.parquet"))
print(f"Nodes: {len(nodes):,}")

nodes = nodes.copy()
nodes["age_days"]           = nodes["age_days"].fillna(0).clip(lower=0)
nodes["versions_count"]     = nodes["versions_count"].fillna(0)
nodes["has_stable_version"] = nodes["has_stable_version"].fillna(0).astype(int)
nodes["stars_count"]        = nodes["stars_count"].fillna(0)
nodes["contributors_count"] = nodes["contributors_count"].fillna(0)

nodes["x1_log_age"]          = np.log(nodes["age_days"] + 1)
nodes["x1_log_versions"]     = np.log(nodes["versions_count"] + 1)
nodes["x1_stable"]           = nodes["has_stable_version"]
nodes["x2_kcore"]            = nodes["kcore"].fillna(0)
nodes["x2_clustering"]       = nodes["clustering"].fillna(0)
nodes["x2_betweenness"]      = nodes["betweenness"].fillna(0)
nodes["x3_log_stars"]        = np.log(nodes["stars_count"] + 1)
nodes["x3_log_contributors"] = np.log(nodes["contributors_count"] + 1)
nodes["y"] = np.log(nodes["dependent_projects_count"] + 1)

X1 = ["x1_log_age", "x1_log_versions", "x1_stable"]
X2 = ["x2_kcore", "x2_clustering", "x2_betweenness"]
X3 = ["x3_log_stars", "x3_log_contributors"]

MODELS = {
    "M1_quality":   X1,
    "M2_position":  X1 + X2,
    "M3_community": X1 + X2 + X3,
    "M4_spillover": X1 + X2 + X3 + ["x4_neighbourhood"],
}

# ── neighbourhood lag for M4 (same construction as 06) ────────────────────────
section("Building neighbourhood lag (x4) from W_cooc")
nodes["x4_neighbourhood"] = 0.0
for plat in PLATFORMS:
    w_path = os.path.join(DATA_DIR, f"W_cooc_{plat}.npz")
    if not os.path.exists(w_path):
        print(f"  [{plat}] W_cooc not found — M4 will be skipped for this platform")
        continue
    W      = sparse.load_npz(w_path)
    n_plat = nodes[nodes["platform"] == plat].reset_index(drop=True)
    y      = n_plat["y"].fillna(0).values
    nb     = np.array(W @ y).flatten()
    idx    = {nm: i for i, nm in enumerate(n_plat["name"])}
    nodes.loc[nodes["platform"] == plat, "x4_neighbourhood"] = (
        nodes.loc[nodes["platform"] == plat, "name"]
        .map(lambda nm: nb[idx[nm]] if nm in idx else 0.0).values
    )
    print(f"  [{plat}] neighbourhood lag attached ({(nb > 0).sum():,} non-zero)")


def flag_vif(v):
    return "OK" if v < 5 else ("caution" if v < 10 else "CONCERN")

def flag_cond(c):
    return "OK" if c < 30 else "caution"


# ── Run diagnostics ───────────────────────────────────────────────────────────

section("OLS ASSUMPTION DIAGNOSTICS")
rows = []
resid_store = {}   # for plots: (platform) -> (fitted, resid)

for plat in PLATFORMS:
    print(f"\n{'─'*64}\n  Platform: {plat}\n{'─'*64}")

    df = nodes[nodes["platform"] == plat].copy()
    df = df[df["versions_count"] > 0].copy()

    # standardise within platform (identical to 06)
    all_x = X1 + X2 + X3 + ["x4_neighbourhood"]
    for col in all_x:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0

    # DV zero-mass (specification flag) — reported once per platform
    zero_frac = (df["dependent_projects_count"] == 0).mean()
    print(f"  DV zero-mass: {zero_frac*100:5.1f}%  of packages have 0 dependents")
    if zero_frac > 0.5:
        print(f"    -> heavy zero-inflation; a hurdle / zero-inflated / count model")
        print(f"       may be better specified than OLS on log(y+1). Flag in Limitations.")

    has_w = os.path.exists(os.path.join(DATA_DIR, f"W_cooc_{plat}.npz"))

    for mname, cols in MODELS.items():
        if mname == "M4_spillover" and not has_w:
            continue
        zcols = [f"{c}_z" for c in cols]
        m = smf.ols("y ~ " + " + ".join(zcols), data=df).fit()

        n, k = int(m.nobs), len(zcols)
        exog = m.model.exog                      # includes const
        resid = m.resid.values
        fitted = m.fittedvalues.values

        # A2 multicollinearity: VIF (exclude const at col 0) + condition number
        vifs = {}
        for j, c in enumerate(zcols, start=1):   # col 0 is const
            try:
                vifs[c] = variance_inflation_factor(exog, j)
            except Exception:
                vifs[c] = np.nan
        max_vif = np.nanmax(list(vifs.values())) if vifs else np.nan
        cond_no = float(m.condition_number)

        # A1 linearity: Ramsey RESET (fitted^2, ^3)
        try:
            reset = linear_reset(m, power=[2, 3], use_f=True)
            reset_f, reset_p = float(reset.fvalue), float(reset.pvalue)
        except Exception:
            reset_f, reset_p = np.nan, np.nan

        # A4 heteroskedasticity: Breusch-Pagan (always) + White (optional, subsampled)
        bp_lm, bp_p, _, _ = het_breuschpagan(resid, exog)
        bp_lm_over_df = bp_lm / k if k > 0 else np.nan
        white_lm = white_p = np.nan
        if COMPUTE_WHITE:
            if n > WHITE_MAX_N:
                rng = np.random.default_rng(SEED)
                sel = rng.choice(n, WHITE_MAX_N, replace=False)
                try:
                    white_lm, white_p, _, _ = het_white(resid[sel], exog[sel])
                except Exception:
                    pass
            else:
                try:
                    white_lm, white_p, _, _ = het_white(resid, exog)
                except Exception:
                    pass

        # A5 normality: Jarque-Bera (+ skew, kurtosis)
        jb, jb_p, skew, kurt = jarque_bera(resid)
        excess_kurt = kurt - 3.0

        # A3 spatial dependence on residuals (M3, M4 only; needs W)
        moran_I = moran_z = moran_p = np.nan
        lm_err = lm_err_p = lm_lag = lm_lag_p = np.nan
        n_conn = np.nan
        if has_w and mname in ("M3_community", "M4_spillover"):
            W_full = sparse.load_npz(os.path.join(DATA_DIR, f"W_cooc_{plat}.npz"))
            n_full = nodes[nodes["platform"] == plat].reset_index(drop=True)
            fidx   = {nm: i for i, nm in enumerate(n_full["name"])}
            keep   = [fidx[nm] for nm in df["name"].values if nm in fidx]
            if len(keep) == len(df):
                Ws   = W_full[keep][:, keep].tocsr()
                deg  = np.asarray((Ws > 0).sum(1)).ravel()
                conn = np.where(deg > 0)[0]
                if len(conn) >= 50:
                    Wc = Ws[conn][:, conn].tocsr()
                    rc = np.asarray(Wc.sum(1)).ravel(); rc[rc == 0] = 1
                    Wc = sparse.diags(1.0 / rc) @ Wc
                    n_conn = int(len(conn))
                    moran_I, _, moran_z, moran_p = morans_i(resid[conn], Wc)
                    lm_err, lm_err_p, lm_lag, lm_lag_p = lm_tests(
                        resid[conn], df["y"].values[conn], exog[conn],
                        fitted[conn], Wc)

        if mname == "M3_community":
            resid_store[plat] = (fitted, resid)

        # ── print compact block ──
        print(f"\n  {mname}   (N={n:,}, k={k}, R²={m.rsquared:.3f})")
        print(f"    A2 multicollinearity : max VIF={max_vif:6.2f} [{flag_vif(max_vif)}]"
              f"   cond.no={cond_no:8.1f} [{flag_cond(cond_no)}]")
        print(f"    A1 linearity (RESET) : F={reset_f:9.2f}  p={reset_p:.2e}")
        print(f"    A4 heteroskedastic.  : BP LM={bp_lm:11.1f} (LM/df={bp_lm_over_df:7.1f}) p={bp_p:.2e}")
        if COMPUTE_WHITE and np.isfinite(white_lm):
            print(f"                           White LM={white_lm:9.1f} p={white_p:.2e}"
                  f"  (subsampled n={min(n, WHITE_MAX_N):,})")
        print(f"    A5 normality (JB)    : skew={skew:+.3f}  exc.kurt={excess_kurt:+.3f}"
              f"   JB={jb:.1f} p={jb_p:.2e}")
        if np.isfinite(moran_I):
            print(f"    A3 spatial (resid)   : Moran I={moran_I:+.4f} z={moran_z:+.1f} p={moran_p:.2e}"
                  f"  |  LM-err={lm_err:.1f} (p={lm_err_p:.2e})  LM-lag={lm_lag:.1f} (p={lm_lag_p:.2e})"
                  f"  [{n_conn:,} connected]")

        row = {
            "platform": plat, "model": mname, "n": n, "k": k,
            "r2": m.rsquared, "adj_r2": m.rsquared_adj,
            "dv_zero_frac": zero_frac,
            "max_vif": max_vif, "condition_number": cond_no,
            "reset_F": reset_f, "reset_p": reset_p,
            "bp_lm": bp_lm, "bp_lm_over_df": bp_lm_over_df, "bp_p": bp_p,
            "white_lm": white_lm, "white_p": white_p,
            "jb": jb, "jb_p": jb_p, "skew": skew, "excess_kurtosis": excess_kurt,
            "moran_I_resid": moran_I, "moran_z_resid": moran_z, "moran_p_resid": moran_p,
            "lm_error": lm_err, "lm_error_p": lm_err_p,
            "lm_lag": lm_lag, "lm_lag_p": lm_lag_p,
            "n_connected": n_conn,
        }
        for c, v in vifs.items():
            row[f"vif_{c}"] = v
        rows.append(row)


# ── Save ──────────────────────────────────────────────────────────────────────

section("Saving")
out = pd.DataFrame(rows)
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, "ols_assumption_diagnostics.csv")
out.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}  ({len(out)} rows)")


# ── Optional residual plots for M3 (visual A1 / A5 checks) ────────────────────

if SAVE_PLOTS and resid_store:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        rng = np.random.default_rng(SEED)

        # residual vs fitted
        fig, axes = plt.subplots(1, len(resid_store), figsize=(5*len(resid_store), 4))
        if len(resid_store) == 1:
            axes = [axes]
        for ax, (plat, (fit, res)) in zip(axes, resid_store.items()):
            s = rng.choice(len(res), min(PLOT_SAMPLE, len(res)), replace=False)
            ax.scatter(fit[s], res[s], s=5, alpha=0.3, linewidths=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{plat} — M3 residuals vs fitted")
            ax.set_xlabel("Fitted"); ax.set_ylabel("Residual")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "12_resid_vs_fitted.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIG_DIR}/12_resid_vs_fitted.png")

        # QQ plots
        fig, axes = plt.subplots(1, len(resid_store), figsize=(5*len(resid_store), 4))
        if len(resid_store) == 1:
            axes = [axes]
        for ax, (plat, (fit, res)) in zip(axes, resid_store.items()):
            s = rng.choice(len(res), min(PLOT_SAMPLE, len(res)), replace=False)
            probplot(res[s], dist="norm", plot=ax)
            ax.set_title(f"{plat} — M3 residual QQ")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "13_qq_plots.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIG_DIR}/13_qq_plots.png")
    except Exception as ex:
        print(f"  [plots skipped: {ex}]")

print("\nDiagnostics complete.")
print("Reminder: at these sample sizes the p-values reject almost automatically;")
print("interpret max VIF, condition number, skew/kurtosis, and Moran's I by magnitude.")
