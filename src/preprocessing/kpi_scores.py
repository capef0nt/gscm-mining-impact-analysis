"""
Utilities for working with KPI data and computing formative construct scores.

High-level flow:
- Input: site-level or site-period KPI DataFrame (may have multiple rows per site)
- Group by site_id (if needed)
- Validate required KPI columns
- Compute simple formative constructs such as:
    - OE_HARD (objective operational efficiency)
    - SAFETY_PERF (safety performance)

This module is the KPI-side counterpart to construct_scores.py
(which works with survey-based reflective constructs).
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from src.config.model_config import CORE_KPIS, KPI_CATEGORIES


SITE_ID_COL = "site_id"


# ---------------------------------------------------------------------------
# KPI configuration for formative constructs
# ---------------------------------------------------------------------------

# KPIs used in OE_HARD index: operational efficiency
OE_KPIS_HIGH_IS_BETTER: List[str] = [
    "uptime_percent",
    "tons_per_hour",
]

OE_KPIS_LOW_IS_BETTER: List[str] = [
    "cost_per_ton",
    "rework_rate_percent",
    "energy_kwh_per_ton",
    "water_m3_per_ton",
    "maintenance_cost_per_ton",
]

# KPIs used in SAFETY_PERF index: safety performance
SAFETY_KPIS_HIGH_IS_BETTER: List[str] = [
    "safety_audits_passed_percent",
    "employees_competent_percent",
]

SAFETY_KPIS_LOW_IS_BETTER: List[str] = [
    "ltifr",
    "trifr",
]


# Weights for the "weighted" method (must sum to 1 per index ideally)
OE_WEIGHTS: Dict[str, float] = {
    "uptime_percent": 0.30,
    "tons_per_hour": 0.30,
    "cost_per_ton": 0.20,
    "energy_kwh_per_ton": 0.10,
    "rework_rate_percent": 0.05,
    "maintenance_cost_per_ton": 0.05,
    # water_m3_per_ton is intentionally left out, but could be added
}

SAFETY_WEIGHTS: Dict[str, float] = {
    "ltifr": 0.40,
    "trifr": 0.30,
    "safety_audits_passed_percent": 0.20,
    "employees_competent_percent": 0.10,
}


# ---------------------------------------------------------------------------
# Validation and aggregation
# ---------------------------------------------------------------------------

def validate_kpi_columns(df: pd.DataFrame) -> None:
    """
    Check that the KPI DataFrame contains the site_id column and required KPI columns.

    Raises:
        ValueError if required columns are missing.
    """
    missing: List[str] = []

    if SITE_ID_COL not in df.columns:
        missing.append(SITE_ID_COL)

    required_kpis = set(
        OE_KPIS_HIGH_IS_BETTER
    ) | set(
        OE_KPIS_LOW_IS_BETTER
    ) | set(
        SAFETY_KPIS_HIGH_IS_BETTER
    ) | set(
        SAFETY_KPIS_LOW_IS_BETTER
    )

    for col in required_kpis:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            f"KPI DataFrame is missing required columns: {sorted(set(missing))}"
        )


def aggregate_kpis_per_site(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate KPI data to one row per site.

    If multiple rows per site exist (e.g. time periods), we compute the mean
    of numeric KPIs per site.

    Args:
        df: DataFrame with at least [site_id, KPI columns].

    Returns:
        DataFrame with one row per site_id and numeric KPI means.
    """
    validate_kpi_columns(df)

    grouped = (
        df.groupby(SITE_ID_COL)
        .mean(numeric_only=True)
        .reset_index()
    )

    return grouped


# ---------------------------------------------------------------------------
# Standardisation helpers
# ---------------------------------------------------------------------------

def _standardize_series(s: pd.Series) -> pd.Series:
    """
    Standardize a numeric series to z-scores: (x - mean) / std.

    If std is 0 (constant series), returns 0 for all values to avoid NaNs.
    """
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


# ---------------------------------------------------------------------------
# Formative construct computation
# ---------------------------------------------------------------------------

def compute_oe_hard(df: pd.DataFrame, method: str = "simple") -> pd.Series:
    """
    Compute OE_HARD (objective operational efficiency index) per site.

    method:
        - "simple": equal-weighted average of standardized components
        - "weighted": uses OE_WEIGHTS dictionary
    """
    # Make a copy of needed columns
    tmp = df.copy()

    components: Dict[str, pd.Series] = {}

    # High-is-better KPIs (standardized directly)
    for kpi in OE_KPIS_HIGH_IS_BETTER:
        components[kpi] = _standardize_series(tmp[kpi])

    # Low-is-better KPIs (standardized and sign-flipped)
    for kpi in OE_KPIS_LOW_IS_BETTER:
        components[kpi] = -_standardize_series(tmp[kpi])

    if method == "simple":
        # Equal weights across all available components
        comp_df = pd.DataFrame(components)
        oe_hard = comp_df.mean(axis=1)
    elif method == "weighted":
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for kpi, series in components.items():
            w = OE_WEIGHTS.get(kpi, 0.0)
            if w == 0.0:
                continue
            weighted_sum += w * series
            total_weight += w

        if total_weight == 0:
            # Fallback: avoid division by zero
            oe_hard = weighted_sum
        else:
            oe_hard = weighted_sum / total_weight
    else:
        raise ValueError(f"Unknown method for OE_HARD: {method}")

    return oe_hard


def compute_safety_perf(df: pd.DataFrame, method: str = "simple") -> pd.Series:
    """
    Compute SAFETY_PERF (safety performance index) per site.

    method:
        - "simple": equal-weighted average of standardized components
        - "weighted": uses SAFETY_WEIGHTS dictionary
    """
    tmp = df.copy()
    components: Dict[str, pd.Series] = {}

    # High-is-better KPIs
    for kpi in SAFETY_KPIS_HIGH_IS_BETTER:
        components[kpi] = _standardize_series(tmp[kpi])

    # Low-is-better KPIs (flip sign)
    for kpi in SAFETY_KPIS_LOW_IS_BETTER:
        components[kpi] = -_standardize_series(tmp[kpi])

    if method == "simple":
        comp_df = pd.DataFrame(components)
        safety_perf = comp_df.mean(axis=1)
    elif method == "weighted":
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for kpi, series in components.items():
            w = SAFETY_WEIGHTS.get(kpi, 0.0)
            if w == 0.0:
                continue
            weighted_sum += w * series
            total_weight += w

        if total_weight == 0:
            safety_perf = weighted_sum
        else:
            safety_perf = weighted_sum / total_weight
    else:
        raise ValueError(f"Unknown method for SAFETY_PERF: {method}")

    return safety_perf


def build_site_kpi_table(
    kpi_df: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    High-level function to:
    - aggregate KPI data to site level
    - compute OE_HARD and SAFETY_PERF indices

    Args:
        kpi_df: raw KPI DataFrame (may have multiple rows per site).
        method: "simple" or "weighted" for index calculation.

    Returns:
        DataFrame with:
            - site_id
            - all numeric KPIs (aggregated)
            - OE_HARD
            - SAFETY_PERF
    """
    site_kpis = aggregate_kpis_per_site(kpi_df).copy()

    # Compute indices
    site_kpis["OE_HARD"] = compute_oe_hard(site_kpis, method=method)
    site_kpis["SAFETY_PERF"] = compute_safety_perf(site_kpis, method=method)

    return site_kpis


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -------------------------------------------------
    # Example: load data/examples/kpis_example.csv
    # -------------------------------------------------
    import os

    EXAMPLE_PATH = os.path.join("data", "examples", "kpis_example.csv")

    if not os.path.exists(EXAMPLE_PATH):
        raise FileNotFoundError(
            f"Example file not found: {EXAMPLE_PATH}\n"
            "Make sure data/examples/kpis_example.csv exists in the repo."
        )

    print(f"\nLoading KPI data from: {EXAMPLE_PATH}\n")

    raw_kpis = pd.read_csv(EXAMPLE_PATH)

    print("--- Raw KPI Data (head) ---")
    print(raw_kpis.head())

    site_kpis_simple = build_site_kpi_table(raw_kpis, method="simple")
    print("\n--- Site KPI Table (simple indices) ---")
    print(site_kpis_simple)

    site_kpis_weighted = build_site_kpi_table(raw_kpis, method="weighted")
    print("\n--- Site KPI Table (weighted indices) ---")
    print(site_kpis_weighted)
