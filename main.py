"""
Entry point for running the GSCM mining impact analysis pipeline.

Pipeline (real-data mode):
1. Load survey data (Likert responses)
2. Compute site-level construct scores (reflective constructs)
3. Load KPI data (objective metrics)
4. Compute site-level KPI indices (formative constructs)
5. Merge into a single site-level dataset
6. Compute correlation table

In synthetic mode:
- Skip steps 1–5 and instead load:
    data/outputs/site_level_synthetic.csv
- Then compute the same correlation table on synthetic data.
"""

from __future__ import annotations

import os
import pandas as pd

from src.preprocessing.construct_scores import build_site_construct_table
from src.preprocessing.kpi_scores import build_site_kpi_table
from src.analysis.correlations import correlation_table
import matplotlib.pyplot as plt  # currently unused, but kept if you plot later


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SURVEY_PATH = os.path.join(BASE_DIR, "data", "examples", "survey_example.csv")
KPI_PATH = os.path.join(BASE_DIR, "data", "examples", "kpis_example.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "site_level_merged.csv")
SYNTHETIC_PATH = os.path.join(OUTPUT_DIR, "site_level_synthetic.csv")


def load_csv_or_raise(path: str, label: str) -> pd.DataFrame:
    """Small helper to load a CSV with a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} file not found at: {path}\n"
            f"Make sure the file exists or adjust the path in main.py."
        )
    return pd.read_csv(path)


def main(method: str = "simple", use_synthetic: bool = False) -> None:
    """
    Run the full preprocessing pipeline or load synthetic site-level data.

    Parameters
    ----------
    method : {"simple", "weighted"}
        How to compute KPI indices in real-data mode.
    use_synthetic : bool
        If True, skip survey/KPI processing and load
        data/outputs/site_level_synthetic.csv instead.
    """
    print("\n=== GSCM Mining Impact Analysis: Site-Level Pipeline ===\n")

    if use_synthetic:
        # -------------------------------------------------------------------
        # SYNTHETIC MODE: just load the synthetic site-level dataset
        # -------------------------------------------------------------------
        print("Running in SYNTHETIC mode.")
        print(f"Loading synthetic site-level data from: {SYNTHETIC_PATH}")
        merged = load_csv_or_raise(SYNTHETIC_PATH, "Synthetic site-level")

        print("\n--- Synthetic Site-Level Dataset (head) ---")
        print(merged.head())

    else:
        # -------------------------------------------------------------------
        # REAL-DATA MODE: build merged dataset from survey + KPI inputs
        # -------------------------------------------------------------------
        print("Running in REAL-DATA mode.")

        # 1) Load survey data
        print(f"\nLoading survey data from: {SURVEY_PATH}")
        survey_df = load_csv_or_raise(SURVEY_PATH, "Survey")
        print(f"Survey data loaded with shape: {survey_df.shape}")

        # 2) Compute site-level construct scores
        print("Computing site-level construct scores (reflective)...")
        site_constructs = build_site_construct_table(survey_df)
        print("\n--- Site Construct Scores (head) ---")
        print(site_constructs.head())
        print("Survey constructs site_ids:", site_constructs["site_id"].unique())

        # 3) Load KPI data
        print(f"\nLoading KPI data from: {KPI_PATH}")
        kpi_df = load_csv_or_raise(KPI_PATH, "KPI")
        print(f"KPI data loaded with shape: {kpi_df.shape}")

        # 4) Compute site-level KPI indices
        print(f"Computing site-level KPI indices (method='{method}')...")
        site_kpis = build_site_kpi_table(kpi_df, method=method)
        print("\n--- Site KPI Table (head) ---")
        print(site_kpis.head())

        # Quick check of the dataset before merging
        print("\nQuick site_id check before merge:")
        print("Survey constructs site_ids:", site_constructs["site_id"].unique())
        print("KPI site_ids:", site_kpis["site_id"].unique())

        # 5) Merge on site_id
        print("\nMerging constructs and KPIs on 'site_id'...")
        merged = pd.merge(site_constructs, site_kpis, on="site_id", how="inner")

        print("\n--- Merged Site-Level Dataset (head) ---")
        print(merged.head())

        # 6) Save merged output (real-data merged file)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        merged.to_csv(MERGED_OUTPUT_PATH, index=False)
        print(f"\nMerged dataset saved to: {MERGED_OUTPUT_PATH}\n")

    # -----------------------------------------------------------------------
    # Correlation analysis (same for real or synthetic)
    # -----------------------------------------------------------------------
    print("\n--- Correlation Matrix ---")
    print(correlation_table(merged))


if __name__ == "__main__":
    # Change use_synthetic=True when you want to run the analysis on
    # site_level_synthetic.csv instead of rebuilding from survey/KPIs.
    main(method="weighted", use_synthetic=True)
