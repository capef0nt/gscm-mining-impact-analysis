from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _ols_regression(y: pd.Series, X: pd.DataFrame) -> Dict:
    """
    Simple OLS using numpy.linalg.lstsq.
    Adds an intercept term automatically.

    Returns:
        dict with coefficients, R2, and column names.
    """
    # Drop rows with NaN in any relevant column
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0].values
    X_clean = data.iloc[:, 1:].values

    # Add intercept
    X_design = np.column_stack([np.ones(len(X_clean)), X_clean])

    coef, residuals, rank, s = np.linalg.lstsq(X_design, y_clean, rcond=None)

    y_pred = X_design @ coef
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "intercept": coef[0],
        "coefficients": dict(zip(X.columns, coef[1:])),
        "r2": r2,
        "n": len(y_clean),
    }


def run_structural_paths(df: pd.DataFrame) -> None:
    """
    Run a set of SEM-like path regressions on the site-level dataset
    and print results in a readable form.
    """
    path_specs: Dict[str, List[str]] = {
        # Mediators → OE
        "OE": ["MAINT", "COMP", "SUPINT"],
        # OE → EP
        "EP": ["OE"],
        # Constructs + mediators → OE_HARD
        "OE_HARD": ["MAINT", "SUPINT", "COMP", "GPUR", "GOPS", "GLOG", "GTRN", "GCOL"],
        # Safety drivers → SAFETY_PERF
        "SAFETY_PERF": ["GTRN", "MAINT", "COMP"],
    }

    for target, predictors in path_specs.items():
        print(f"\n=== Path model: {target} ~ {', '.join(predictors)} ===")
        y = df[target]
        X = df[predictors]

        results = _ols_regression(y, X)

        print(f"n = {results['n']}, R² = {results['r2']:.3f}")
        print("Intercept:", round(results["intercept"], 3))
        print("Coefficients:")
        for name, coef in results["coefficients"].items():
            print(f"  {name}: {coef:.3f}")
