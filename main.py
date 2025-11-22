"""
Entry point for running the GSCM mining impact analysis pipeline.

Pipeline:
1. Load survey data (Likert responses)
2. Compute site-level construct scores (reflective constructs)
3. Load KPI data (objective metrics)
4. Compute site-level KPI indices (formative constructs)
5. Merge into a single site-level dataset
"""

from __future__ import annotations

import os
import pandas as pd

from src.preprocessing.construct_scores import build_site_construct_table
from src.preprocessing.kpi_scores import build_site_kpi_table


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SURVEY_PATH = os.path.join(BASE_DIR, "data", "examples", "survey_example.csv")
KPI_PATH = os.path.join(BASE_DIR, "data", "examples", "kpis_example.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "site_level_merged.csv")


def load_csv_or_raise(path: str, label: str) -> pd.DataFrame:
    """Small helper to load a CSV with a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} file not found at: {path}\n"
            f"Make sure the file exists or adjust the path in main.py."
        )
    return pd.read_csv(path)


def main(method: str = "simple") -> None:
    """
    Run the full preprocessing pipeline:

    - method: how to compute KPI indices ("simple" or "weighted")
    """
    print("\n=== GSCM Mining Impact Analysis: Site-Level Merge ===\n")

    # 1) Load survey data
    print(f"Loading survey data from: {SURVEY_PATH}")
    survey_df = load_csv_or_raise(SURVEY_PATH, "Survey")

    # 2) Compute site-level construct scores
    print("Computing site-level construct scores (reflective)...")
    site_constructs = build_site_construct_table(survey_df)
    print("\n--- Site Construct Scores (head) ---")
    print(site_constructs.head())

    # 3) Load KPI data
    print(f"\nLoading KPI data from: {KPI_PATH}")
    kpi_df = load_csv_or_raise(KPI_PATH, "KPI")

    # 4) Compute site-level KPI indices
    print(f"Computing site-level KPI indices (method='{method}')...")
    site_kpis = build_site_kpi_table(kpi_df, method=method)
    print("\n--- Site KPI Table (head) ---")
    print(site_kpis.head())

    # 5) Merge on site_id
    print("\nMerging constructs and KPIs on 'site_id'...")
    merged = pd.merge(site_constructs, site_kpis, on="site_id", how="inner")

    print("\n--- Merged Site-Level Dataset (head) ---")
    print(merged.head())

    # 6) Save merged output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"\nMerged dataset saved to: {MERGED_OUTPUT_PATH}\n")


if __name__ == "__main__":
    # Change to method="weighted" if you want weighted indices by default
    main(method="simple")
