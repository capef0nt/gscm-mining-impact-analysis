"""
Outer (measurement) model utilities for reflective constructs.

For each construct in src.config.model_config.CONSTRUCTS, this module can:
- Compute Cronbach's alpha (internal consistency)
- Compute composite reliability (CR)
- Compute AVE (Average Variance Extracted)
- Compute approximate indicator loadings (correlation of each indicator with
  the construct composite)

Assumptions:
- All constructs are reflective.
- The DataFrame provided has one column per survey indicator, using the names
  specified in model_config.CONSTRUCTS[code].indicators.

This is a simplified, educational approximation of PLS-SEM outer model logic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.model_config import CONSTRUCTS, ConstructConfig


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _cronbach_alpha(X: pd.DataFrame) -> float:
    """
    Compute Cronbach's alpha for a set of items (columns) in X.

    Alpha = (k / (k - 1)) * (1 - sum(var_i) / var(total_score))

    Returns NaN if there are fewer than 2 items or total variance is zero.
    """
    k = X.shape[1]
    if k < 2:
        return np.nan

    # Use sample variance (ddof=1) for items and total
    item_vars = X.var(ddof=1)
    total_score = X.sum(axis=1)
    total_var = total_score.var(ddof=1)

    if total_var == 0 or np.isnan(total_var):
        return np.nan

    alpha = (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)
    return float(alpha)


def _compute_loadings_cr_ave(X: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """
    Compute indicator loadings, composite reliability (CR) and AVE for a
    construct, given a DataFrame of its indicators (columns = items).

    Approach:
    - Standardize indicators.
    - Define construct composite as the mean of standardized indicators.
    - Loading for each indicator = Pearson correlation with the composite.
    - Assume measurement error variance θ_i = 1 - λ_i^2 (standardized).
    - CR = ( (Σ λ_i)^2 ) / ( (Σ λ_i)^2 + Σ θ_i )
    - AVE = Σ λ_i^2 / k
    """
    if X.shape[1] == 0:
        return {}, np.nan, np.nan

    # Standardize indicators
    Z = (X - X.mean()) / X.std(ddof=0)
    # If any column has zero variance, std will be 0 → replace with 0
    Z = Z.fillna(0.0)

    # Construct composite = mean of standardized indicators
    composite = Z.mean(axis=1)

    loadings: Dict[str, float] = {}
    for col in Z.columns:
        x = Z[col]
        # Pearson correlation with the composite; guard for degenerate case
        if x.std(ddof=0) == 0 or composite.std(ddof=0) == 0:
            l = np.nan
        else:
            l = float(x.corr(composite))
        loadings[col] = l

    # Prepare λ (loadings) for CR and AVE (drop NaN)
    lambdas = np.array([v for v in loadings.values() if not np.isnan(v)])
    if lambdas.size == 0:
        return loadings, np.nan, np.nan

    theta = 1.0 - lambdas**2  # error variances (assuming standardized indicators)

    num = (lambdas.sum())**2
    den = num + theta.sum()
    cr = float(num / den) if den != 0 else np.nan
    ave = float((lambdas**2).mean())

    return loadings, cr, ave


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_outer_model(
    df: pd.DataFrame,
    construct_codes: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compute outer-model statistics for all (or selected) reflective constructs.

    Args:
        df:
            DataFrame containing one column per survey indicator. Columns must
            match the indicator names in model_config.CONSTRUCTS.
        construct_codes:
            Optional list of construct codes to restrict the analysis. If None,
            all constructs in CONSTRUCTS are processed.

    Returns:
        DataFrame where each row is a construct with:
            - construct
            - name
            - n_indicators
            - n_obs
            - alpha
            - cr
            - ave
            - loading_<indicator_name> for each indicator
    """
    if construct_codes is None:
        construct_codes = list(CONSTRUCTS.keys())

    results: List[Dict[str, float]] = []

    for code in construct_codes:
        cfg: ConstructConfig = CONSTRUCTS[code]

        # Select indicator columns that are present in df
        indicator_cols = [col for col in cfg.indicators if col in df.columns]
        if len(indicator_cols) < 2:
            # Need at least 2 items for alpha / CR / AVE
            # We still could compute loadings with 1 item, but it's not
            # meaningful as a reflective construct. Skip or log.
            continue

        X = df[indicator_cols].dropna()
        if X.empty:
            continue

        alpha = _cronbach_alpha(X)
        loadings, cr, ave = _compute_loadings_cr_ave(X)

        row: Dict[str, float] = {
            "construct": code,
            "name": cfg.name,
            "n_indicators": len(indicator_cols),
            "n_obs": len(X),
            "alpha": alpha,
            "cr": cr,
            "ave": ave,
        }

        # Add loadings under loading_<indicator_name>
        for ind_name, lval in loadings.items():
            row[f"loading_{ind_name}"] = lval

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Small demo using data/examples/survey_example.csv, if it exists.
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    EXAMPLE_PATH = os.path.join(BASE_DIR, "data", "examples", "survey_example.csv")

    if not os.path.exists(EXAMPLE_PATH):
        raise FileNotFoundError(
            f"Example file not found: {EXAMPLE_PATH}\n"
            "Make sure data/examples/survey_example.csv exists."
        )

    print(f"Loading example survey data from: {EXAMPLE_PATH}")
    demo_df = pd.read_csv(EXAMPLE_PATH)

    outer_stats = compute_outer_model(demo_df)
    print("\n--- Outer model stats (demo) ---")
    print(outer_stats)
