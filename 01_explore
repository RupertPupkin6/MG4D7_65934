import pandas as pd
import os

DATA_DIR = "data"

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def explore(filename, nrows=None):
    path = os.path.join(DATA_DIR, filename)
    section(filename)

    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    print(f"\nShape:   {df.shape[0]:>10,} rows  x  {df.shape[1]} columns")
    print(f"\nColumns:\n  {list(df.columns)}")

    # Null counts
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"\nNulls:")
        for col, n in nulls.items():
            pct = n / len(df) * 100
            print(f"  {col:<45} {n:>8,}  ({pct:.1f}%)")
    else:
        print("\nNulls: none")

    # Platform distribution if column exists
    if "platform" in df.columns:
        print(f"\nPlatform distribution:")
        counts = df["platform"].value_counts()
        for plat, n in counts.items():
            pct = n / len(df) * 100
            print(f"  {plat:<20} {n:>10,}  ({pct:.1f}%)")

    # Sample rows
    print(f"\nSample (5 rows):")
    print(df.head(5).to_string())

    return df

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load full projects (small enough)
    projects = explore("projects.csv")

    # Load sample of large files to avoid memory issues
    deps     = explore("dependencies.csv", nrows=500_000)
    versions = explore("versions.csv",     nrows=500_000)
    repos    = explore("repositories.csv")

    # ── Cross-file checks ─────────────────────────────────────────────────────
    section("Cross-file checks")

    # How many unique packages in dependencies vs projects?
    print(f"\nUnique packages in dependencies (project_name x platform):")
    dep_packages = deps.groupby("platform")["project_name"].nunique()
    for plat, n in dep_packages.items():
        print(f"  {plat:<20} {n:>10,} unique packages")

    print(f"\nUnique packages in projects:")
    proj_packages = projects.groupby("platform")["name"].nunique()
    for plat, n in proj_packages.items():
        print(f"  {plat:<20} {n:>10,} unique packages")

    # Adoption metric distribution per platform
    section("Adoption metrics (dependent_projects_count) per platform")
    for plat in ["NPM", "Pypi", "Packagist"]:
        sub = projects[projects["platform"] == plat]["dependent_projects_count"]
        print(f"\n{plat}:")
        print(f"  count:  {len(sub):>10,}")
        print(f"  mean:   {sub.mean():>10.2f}")
        print(f"  median: {sub.median():>10.2f}")
        print(f"  max:    {sub.max():>10,.0f}")
        print(f"  zeros:  {(sub == 0).sum():>10,}  ({(sub==0).mean()*100:.1f}%)")

    # Dependency kind distribution
    section("Dependency kinds in dependencies.csv (sample)")
    if "dependency_kind" in deps.columns:
        print(deps["dependency_kind"].value_counts().to_string())

    # Status distribution in projects
    section("Package status distribution in projects.csv")
    print(projects["status"].value_counts(dropna=False).to_string())

    print("\n\nExploration complete.")
