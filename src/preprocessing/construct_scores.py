"""
Utilities for computing construct scores from survey data.

High-level flow:
- Input: respondent-level survey DataFrame (one row per person)
- Group by site_id
- Compute mean of each indicator per site
- Compute simple construct scores per site (mean of indicators per construct)

This module relies on model configuration from src.config.model_config.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.config.model_config import CONSTRUCTS, get_construct_codes, all_indicators


SITE_ID_COL = "site_id"


def validate_survey_columns(df: pd.DataFrame) -> None:
    """
    Check that the survey DataFrame contains all required indicator columns
    and the site_id column.

    Raises:
        ValueError if required columns are missing.
    """
    missing: List[str] = []

    # site_id must be present
    if SITE_ID_COL not in df.columns:
        missing.append(SITE_ID_COL)

    # all indicator columns from config must be present
    for col in all_indicators():
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            f"Survey DataFrame is missing required columns: {sorted(set(missing))}"
        )


def compute_indicator_means_per_site(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate respondent-level survey data to site-level indicator means.

    Args:
        df: DataFrame with one row per respondent, including site_id and all indicators.

    Returns:
        DataFrame with one row per site_id and one column per indicator (mean value).
    """
    # Ensure required columns exist
    validate_survey_columns(df)

    # Group by site_id and compute the mean of all numeric columns (Likert items).
    grouped = (
        df.groupby(SITE_ID_COL)
        .mean(numeric_only=True)
        .reset_index()
    )

    # Keep only site_id + indicator columns (ignore other numeric fields, if any)
    indicator_cols: List[str] = all_indicators()
    cols_to_keep = [SITE_ID_COL] + [c for c in indicator_cols if c in grouped.columns]

    return grouped[cols_to_keep]


def compute_construct_scores(indicator_means: pd.DataFrame) -> pd.DataFrame:
    """
    Compute construct scores per site from indicator-level means.

    For now, construct score = mean of its indicators for that site.

    Args:
        indicator_means: DataFrame with columns [site_id, indicator1, indicator2, ...]

    Returns:
        DataFrame with columns [site_id, GPUR, GOPS, ..., EP]
    """
    # Start with site_id column
    site_id = indicator_means[SITE_ID_COL]
    construct_scores = pd.DataFrame({SITE_ID_COL: site_id})

    for code in get_construct_codes():
        cfg = CONSTRUCTS[code]
        # Only keep indicators that are actually present in the DataFrame
        indicators = [col for col in cfg.indicators if col in indicator_means.columns]

        if not indicators:
            # If no indicators available (should not happen if validation passed),
            # we skip this construct.
            continue

        # Simple composite: mean of indicator means for this construct.
        construct_scores[code] = indicator_means[indicators].mean(axis=1)

    # Remove any accidental duplicate site_id columns
    construct_scores = construct_scores.loc[:, ~construct_scores.columns.duplicated()]

    return construct_scores


def build_site_construct_table(survey_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to:
    - validate survey data
    - compute indicator means per site
    - compute construct scores per site

    Args:
        survey_df: respondent-level survey DataFrame.

    Returns:
        DataFrame with one row per site_id and one column per construct code.
    """
    indicator_means = compute_indicator_means_per_site(survey_df)
    construct_scores = compute_construct_scores(indicator_means)
    return construct_scores


if __name__ == "__main__":
    # -------------------------------------------------
    # Example usage: load data/examples/survey_example.csv
    # -------------------------------------------------
    import os

    EXAMPLE_PATH = os.path.join("data", "examples", "survey_example.csv")

    if not os.path.exists(EXAMPLE_PATH):
        raise FileNotFoundError(
            f"Example file not found: {EXAMPLE_PATH}\n"
            "Make sure data/examples/survey_example.csv exists in the repo."
        )

    print(f"\nLoading survey data from: {EXAMPLE_PATH}\n")

    survey_df = pd.read_csv(EXAMPLE_PATH)

    print("--- Raw Survey Data (head) ---")
    print(survey_df.head())

    # Build construct scores per site
    site_constructs = build_site_construct_table(survey_df)

    print("\n--- Construct Scores (Per Site) ---")
    print(site_constructs)
