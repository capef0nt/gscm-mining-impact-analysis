from __future__ import annotations

import os

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# Adjust BASE_DIR depth if your project layout changes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "outputs", "site_level_synthetic.csv")


def main() -> None:
    print(f"Loading synthetic site-level data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # === 1. Basic sanity checks ===
    print("\n=== Head ===")
    print(df.head())

    print("\n=== Describe ===")
    print(df.describe().T.round(3))

    print("\n=== Correlation matrix (numeric columns only) ===")
    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.corr().round(2))

    # === 2. Define GSCM predictors and outcome variables ===
    # Adjust these names if needed to match your actual columns
    gscm_cols = ["GPUR", "GOPS", "GLOG", "GTRN"]
    outcome_cols = [
        "frontline_stoppages_percent",
        "OE_HARD",
        "SAFETY_PERF",
    ]

    for col in gscm_cols + outcome_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in synthetic dataset.")

    X = df[gscm_cols].copy()
    Y = df[outcome_cols].copy()

    # Drop any rows with missing values (synthetic should be clean, but just in case)
    data = pd.concat([X, Y], axis=1).dropna()
    X = data[gscm_cols]
    Y = data[outcome_cols]

    print(f"\nUsing {len(X)} rows after dropping NA.")

    # === 3. Standardise X and Y ===
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)

    # Number of components – min(#predictors, #outcomes, n_samples-1)
    n_components = min(len(gscm_cols), len(outcome_cols), len(X) - 1)
    print(f"\nFitting PLSRegression with {n_components} components...")
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    Y_pred_scaled = pls.predict(X_scaled)
    Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)

    # === 4. Evaluate: R² per outcome ===
    print("\n=== R² by outcome ===")
    for i, col in enumerate(outcome_cols):
        r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
        print(f"{col}: R² = {r2:.3f}")

    # === 5. Inspect loadings / coefficients ===
    coefs = pls.coef_  # shape may vary depending on sklearn version

    if coefs.shape == (len(gscm_cols), len(outcome_cols)):
        # predictors as rows, outcomes as columns
        coef_df = pd.DataFrame(
            coefs,
            index=gscm_cols,
            columns=outcome_cols,
        )
    elif coefs.shape == (len(outcome_cols), len(gscm_cols)):
        # outcomes as rows, predictors as columns -> transpose
        coef_df = pd.DataFrame(
            coefs,
            index=outcome_cols,
            columns=gscm_cols,
        ).T
    else:
        raise ValueError(
            f"Unexpected coef_ shape: {coefs.shape}. "
            f"Expected ({len(gscm_cols)}, {len(outcome_cols)}) "
            f"or ({len(outcome_cols)}, {len(gscm_cols)})."
        )

    print("\n=== PLS coefficients (standardised space) ===")
    print(coef_df.round(3))

    # Optional: save coefficients for reporting
    output_coef_path = os.path.join(
        BASE_DIR, "data", "outputs", "pls_site_level_coefficients.csv"
    )
    coef_df.to_csv(output_coef_path)
    print(f"\nPLS coefficients saved to: {output_coef_path}")


if __name__ == "__main__":
    main()
