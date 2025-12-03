"""
Generate realistic synthetic respondent-level survey data based on
site-level construct scores.

Input:
    data/outputs/site_level_synthetic.csv
        - One row per site (site_id)
        - Columns include the latent constructs defined in CONSTRUCTS
          (e.g., GPUR, GOPS, GLOG, GTRN, GCOL, SUPINT, MAINT, COMP, OE, EP)

Output:
    data/outputs/survey_synthetic.csv
        - Multiple rows per site (respondents_per_site each)
        - Columns:
            respondent_id, site_id, company_id,
            and all indicator columns in CONSTRUCTS[code].indicators.

Realistic generation logic:
    For each site s and construct c:
        1. Take site-level score μ_sc (from site_level_synthetic).
        2. For each respondent r at site s:
            latent_scr = μ_sc + N(0, latent_sigma)
            indicator_ij = latent_scr + N(0, indicator_sigma)
            then round/clamp to Likert 1..5.

This makes:
    - Items within a construct correlated (same latent per respondent).
    - Different respondents at same site similar but not identical.
    - Different sites can have different mean construct levels.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Path setup so we can import src.config.model_config
# -------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
DATA_GEN_DIR = os.path.dirname(THIS_FILE)                  # .../src/data_generation
SRC_DIR = os.path.dirname(DATA_GEN_DIR)                    # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                    # project root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.model_config import CONSTRUCTS  # noqa: E402


SITE_LEVEL_PATH = os.path.join(
    PROJECT_ROOT, "data", "outputs", "site_level_synthetic.csv"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "outputs", "survey_synthetic.csv"
)


def _likert_from_latent(
    rng: np.random.Generator,
    latent_value: float,
    indicator_sigma: float = 0.5,
) -> int:
    """
    Given a latent construct value (~1..5), generate a Likert item.

    - Add normal noise with std = indicator_sigma
    - Clip to [1, 5]
    - Round to nearest integer
    """
    val = latent_value + rng.normal(loc=0.0, scale=indicator_sigma)
    val = float(np.clip(val, 1.0, 5.0))
    return int(np.rint(val))


def generate_synthetic_survey(
    respondents_per_site: int = 8,
    latent_sigma: float = 0.4,
    indicator_sigma: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate respondent-level synthetic survey data.

    Args:
        respondents_per_site:
            Number of respondents to simulate for each site.
        latent_sigma:
            Standard deviation of respondent-level latent scores around
            the site-level construct score.
        indicator_sigma:
            Standard deviation of indicator-level noise around the latent score.
        random_seed:
            RNG seed for reproducibility.

    Returns:
        DataFrame with columns:
            respondent_id, site_id, company_id, and all indicator columns.
    """
    rng = np.random.default_rng(random_seed)

    if not os.path.exists(SITE_LEVEL_PATH):
        raise FileNotFoundError(
            f"Site-level synthetic data not found at:\n  {SITE_LEVEL_PATH}\n"
            "Run your site-level synthetic generation first "
            "(e.g., scripts/generate_site_level_synthetic.py)."
        )

    site_df = pd.read_csv(SITE_LEVEL_PATH)

    if "site_id" not in site_df.columns:
        raise ValueError(
            "Expected 'site_id' column in site-level synthetic data, "
            "but it was not found."
        )

    rows: List[Dict] = []

    # Iterate over each synthetic site
    for _, site_row in site_df.iterrows():
        site_id = site_row["site_id"]
        # Simple placeholder company_id; can later be replaced with real mapping.
        company_id = site_row.get("company_id", "SyntheticCo")

        for r in range(respondents_per_site):
            respondent_id = f"{site_id}_R{r+1}"
            row: Dict = {
                "respondent_id": respondent_id,
                "site_id": site_id,
                "company_id": company_id,
            }

            # For every construct in the config, generate indicators
            for code, cfg in CONSTRUCTS.items():
                # Try to pull site-level construct score; if missing, draw fallback
                site_construct_score = site_row.get(code, np.nan)

                if not np.isfinite(site_construct_score):
                    # Fallback: pick a plausible "mid-high" Likert region
                    site_construct_score = rng.uniform(2.5, 4.0)

                # Respondent-specific latent score around the site mean
                latent_resp = site_construct_score + rng.normal(
                    loc=0.0, scale=latent_sigma
                )
                latent_resp = float(np.clip(latent_resp, 1.0, 5.0))

                # Generate indicators for this construct
                for ind in cfg.indicators:
                    row[ind] = _likert_from_latent(
                        rng,
                        latent_resp,
                        indicator_sigma=indicator_sigma,
                    )

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading site-level data from: {SITE_LEVEL_PATH}")

    df = generate_synthetic_survey(
        respondents_per_site=8,
        latent_sigma=0.4,
        indicator_sigma=0.5,
        random_seed=42,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Synthetic survey saved to: {OUTPUT_PATH}")
    print(f"\nNumber of rows (respondents): {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\n--- Survey head ---")
    print(df.head())

    # Quick sanity check: variance of one construct's indicators
    gpur_cols = [c for c in df.columns if c.startswith("GPUR_")]
    if gpur_cols:
        print("\nGPUR indicator variances (sanity check):")
        print(df[gpur_cols].var())


if __name__ == "__main__":
    main()
