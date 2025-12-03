"""
Generate synthetic respondent-level survey data based on site-level constructs.

Input:
    data/outputs/site_level_synthetic.csv
        - One row per site (e.g. SYN_001, SYN_002, ...)
        - Includes latent construct scores like:
            GPUR, GOPS, GLOG, GTRN, GCOL,
            SUPINT, MAINT, COMP,
            OE, EP
        - Plus KPI columns (which we ignore for survey generation).

Output:
    data/outputs/survey_synthetic.csv
        - Multiple rows per site (respondents_per_site each)
        - Columns:
            respondent_id, site_id, company_id,
            GPUR_1.., GOPS_1.., GLOG_1.., ... OE_1.., EP_1..

Design:
- For each site:
    - For each construct in model_config.CONSTRUCTS:
        - Try to use the site-level construct score as latent center.
        - If missing, draw a reasonable latent score (2.5–4.0).
    - For each indicator of that construct:
        - Generate a noisy Likert response (1–5) around the latent center.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Ensure PROJECT_ROOT is on sys.path so `src.*` imports work
# -------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(THIS_FILE)              # .../src/data_generation
PROJECT_ROOT = os.path.dirname(SRC_DIR)           # .../src
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)      # project root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.model_config import CONSTRUCTS, ConstructConfig  # noqa: E402


SITE_LEVEL_PATH = os.path.join(
    PROJECT_ROOT, "data", "outputs", "site_level_synthetic.csv"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "outputs", "survey_synthetic.csv"
)


def _sample_likert_around(
    rng: np.random.Generator,
    center: float,
    scale: float = 0.6,
) -> int:
    """
    Draw a single Likert response (1–5) around a given latent center.

    Args:
        rng: numpy random generator
        center: latent score around which responses cluster (~1–5)
        scale: standard deviation of the noise

    Returns:
        Integer in {1, 2, 3, 4, 5}
    """
    val = rng.normal(loc=center, scale=scale)
    val = np.clip(val, 1.0, 5.0)
    return int(np.rint(val))


def generate_synthetic_survey(
    respondents_per_site: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate respondent-level synthetic survey data.

    For each site in site_level_synthetic.csv:
        - Create `respondents_per_site` respondents.
        - For each construct in CONSTRUCTS:
            - Use site-level construct score if present,
              else draw a fallback latent score.
            - Generate all indicator items using a Likert distribution.

    Returns:
        DataFrame with columns:
            respondent_id, site_id, company_id, and all indicator columns.
    """
    rng = np.random.default_rng(random_seed)

    if not os.path.exists(SITE_LEVEL_PATH):
        raise FileNotFoundError(
            f"Site-level synthetic data not found at:\n  {SITE_LEVEL_PATH}\n"
            "Run your site-level synthetic generation first "
            "(e.g., generate_synthetic_sites_realistic.py)."
        )

    site_df = pd.read_csv(SITE_LEVEL_PATH)

    if "site_id" not in site_df.columns:
        raise ValueError(
            "Expected 'site_id' column in site-level synthetic data, "
            "but it was not found."
        )

    rows: List[Dict] = []

    for _, site in site_df.iterrows():
        site_id = site["site_id"]

        # Create multiple respondents per site
        for i in range(respondents_per_site):
            row: Dict = {
                "respondent_id": f"{site_id}_R{i+1}",
                "site_id": site_id,
                "company_id": "SyntheticCo",  # placeholder; can be extended
            }

            # For every construct configured in the model
            for code, cfg in CONSTRUCTS.items():
                # Try to pull site-level construct score; fall back if missing
                site_construct_score = site.get(code, np.nan)

                if not np.isfinite(site_construct_score):
                    # If SDV or merging didn't include this construct,
                    # we still want it to exist for outer-model analysis.
                    # Draw a reasonable value in the typical Likert "good-ish" range.
                    site_construct_score = rng.uniform(2.5, 4.0)

                # Generate all indicators for this construct
                for indicator in cfg.indicators:
                    row[indicator] = _sample_likert_around(
                        rng,
                        center=float(site_construct_score),
                        scale=0.6,  # controls within-construct variability
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def main(
    respondents_per_site: int = 5,
    random_seed: int = 42,
) -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading site-level data from: {SITE_LEVEL_PATH}")
    print(
        f"Generating synthetic survey with "
        f"{respondents_per_site} respondents per site..."
    )

    df = generate_synthetic_survey(
        respondents_per_site=respondents_per_site,
        random_seed=random_seed,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Synthetic survey saved to: {OUTPUT_PATH}")
    print(f"\nNumber of rows (respondents): {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\n--- Survey head ---")
    print(df.head())


if __name__ == "__main__":
    main(respondents_per_site=5, random_seed=42)
