import os
import re
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR  = os.path.join(DATA_DIR, "prepared")
PLATFORMS = ["NPM", "Pypi", "Packagist"]

os.makedirs(OUT_DIR, exist_ok=True)

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── Step 1: Load and clean projects ──────────────────────────────────────────

section("Step 1: Load and clean projects")

proj = pd.read_csv(
    os.path.join(DATA_DIR, "projects.csv"),
    low_memory=False,
    parse_dates=["latest_release_published_at", "first_published_at"]
)

# Drop useless columns
proj.drop(columns=["status", "language"], inplace=True)

# Drop the 3 rows with null name
proj.dropna(subset=["name"], inplace=True)

# Compute package age in days (snapshot ~2020-01-01)
SNAPSHOT_DATE = pd.Timestamp("2020-01-01", tz="UTC")
proj["age_days"] = (SNAPSHOT_DATE - proj["first_published_at"]).dt.days.clip(lower=0)

# Log-transform adoption metrics
proj["log_dependent_projects"] = np.log(proj["dependent_projects_count"] + 1)
proj["log_dependent_repos"]    = np.log(proj["dependent_repositories_count"] + 1)

print(f"Projects after cleaning: {len(proj):,} rows")
print(f"Platforms: {proj['platform'].value_counts().to_dict()}")

# ── Step 2: Load and clean dependencies ──────────────────────────────────────

section("Step 2: Load and clean dependencies")

deps = pd.read_csv(
    os.path.join(DATA_DIR, "dependencies.csv"),
    low_memory=False,
    usecols=["platform", "project_name", "project_id",
             "dependency_name", "optional_dependency"]
)

# Drop nulls in key columns
deps.dropna(subset=["project_name", "dependency_name"], inplace=True)

# Drop optional dependencies
deps = deps[deps["optional_dependency"] == False].copy()
deps.drop(columns=["optional_dependency"], inplace=True)

# Drop self-loops
deps = deps[deps["project_name"] != deps["dependency_name"]].copy()

print(f"Dependencies after cleaning: {len(deps):,} rows")
print(f"Platforms: {deps['platform'].value_counts().to_dict()}")

# ── Step 3: Latest stable version per package ─────────────────────────────────

section("Step 3: Identify latest stable version per package")

vers = pd.read_csv(
    os.path.join(DATA_DIR, "versions.csv"),
    low_memory=False,
    parse_dates=["published_at"]
)

vers.dropna(subset=["project_name", "version_number"], inplace=True)

# Flag prereleases
prerelease_pattern = r"alpha|beta|rc|dev|pre|snapshot|nightly"
vers["is_prerelease"] = vers["version_number"].astype(str).str.contains(
    prerelease_pattern, case=False, regex=True)

stable = vers[~vers["is_prerelease"]].copy()

latest = (
    stable.sort_values("published_at", ascending=False)
          .groupby(["platform", "project_id"], as_index=False)
          .first()[["platform", "project_id", "project_name",
                    "version_number", "published_at"]]
)
latest.rename(columns={"published_at": "latest_stable_published_at"}, inplace=True)

print(f"Packages with at least one stable version: {len(latest):,}")
print(f"Platforms: {latest['platform'].value_counts().to_dict()}")

# ── Step 4: Join repositories to projects ────────────────────────────────────

section("Step 4: Join repository signals to projects")

repos = pd.read_csv(
    os.path.join(DATA_DIR, "repositories.csv"),
    low_memory=False,
    usecols=["repository_id", "stars_count", "forks_count",
             "contributors_count", "watchers_count"]
)

proj = proj.merge(repos, on="repository_id", how="left")
proj["log_stars"] = np.log(proj["stars_count"].fillna(0) + 1)

for col in ["stars_count", "forks_count", "contributors_count", "watchers_count"]:
    proj[col] = proj[col].fillna(0).astype(int)

print(f"Projects with linked repository: "
      f"{proj['stars_count'].gt(0).sum():,} "
      f"({proj['stars_count'].gt(0).mean()*100:.1f}%)")

# ── Step 5: Build final node table ───────────────────────────────────────────

section("Step 5: Build final node table")

nodes = proj.merge(
    latest[["platform", "project_id", "version_number",
            "latest_stable_published_at"]],
    on=["platform", "project_id"],
    how="left"
)

nodes["has_stable_version"] = nodes["version_number"].notna()

nodes = nodes[[
    "project_id", "platform", "name",
    "dependent_projects_count", "dependent_repositories_count",
    "log_dependent_projects", "log_dependent_repos",
    "sourcerank", "licenses", "versions_count",
    "age_days", "has_stable_version",
    "stars_count", "forks_count", "contributors_count", "watchers_count",
    "log_stars", "repository_id"
]].copy()

print(f"Final node table: {len(nodes):,} rows x {nodes.shape[1]} columns")
for plat in PLATFORMS:
    n = (nodes["platform"] == plat).sum()
    print(f"  {plat:<15} {n:>10,} packages")

# ── Step 6: Build final edge list ────────────────────────────────────────────

section("Step 6: Build final edge list")

# PEP 503 normalisation for PyPI — treats foo-bar, foo_bar, foo.bar as equal
# This recovers ~12x more PyPI edges (from 8k to ~98k)
def normalise_name(name, platform):
    if platform == "Pypi":
        return str(name).lower().replace("-", "_").replace(".", "_")
    return str(name).lower()

# Build canonical name lookup with normalised keys
name_lookup = {}
for _, row in nodes[["platform", "name"]].iterrows():
    key = (row["platform"], normalise_name(row["name"], row["platform"]))
    name_lookup[key] = row["name"]

# Normalise dependency and project names for matching
deps["project_name_norm"] = deps.apply(
    lambda r: normalise_name(r["project_name"], r["platform"]), axis=1)
deps["dependency_name_norm"] = deps.apply(
    lambda r: normalise_name(r["dependency_name"], r["platform"]), axis=1)

# Map back to canonical names
deps["project_name_canon"] = deps.apply(
    lambda r: name_lookup.get((r["platform"], r["project_name_norm"])), axis=1)
deps["dependency_name_canon"] = deps.apply(
    lambda r: name_lookup.get((r["platform"], r["dependency_name_norm"])), axis=1)

# Keep only edges where both endpoints exist
edges = deps.dropna(
    subset=["project_name_canon", "dependency_name_canon"]).copy()
edges["project_name"]    = edges["project_name_canon"]
edges["dependency_name"] = edges["dependency_name_canon"]
edges.drop(columns=[
    "project_name_norm", "dependency_name_norm",
    "project_name_canon", "dependency_name_canon"
], inplace=True)

# Remove self-loops that may appear after normalisation
edges = edges[edges["project_name"] != edges["dependency_name"]].copy()

# Deduplicate
edges.drop_duplicates(
    subset=["platform", "project_name", "dependency_name"], inplace=True)

print(f"Final edge list: {len(edges):,} rows")
for plat in PLATFORMS:
    n = (edges["platform"] == plat).sum()
    print(f"  {plat:<15} {n:>10,} edges")

# ── Save outputs ──────────────────────────────────────────────────────────────

section("Saving outputs")

nodes.to_parquet(os.path.join(OUT_DIR, "nodes.parquet"), index=False)
edges.to_parquet(os.path.join(OUT_DIR, "edges.parquet"), index=False)

print(f"Saved nodes.parquet — {len(nodes):,} rows")
print(f"Saved edges.parquet — {len(edges):,} rows")
print(f"\nLocation: {os.path.abspath(OUT_DIR)}")
print("\nPreparation complete. Ready for graph construction.")
