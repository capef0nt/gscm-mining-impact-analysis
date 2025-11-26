from __future__ import annotations

import os
import sys
import pandas as pd

# -------------------------------------------------------------------
# Make sure we can import `src.*` no matter where we run from
# -------------------------------------------------------------------

# Absolute path to this file: .../gscm-mining-impact-analysis/scripts/run_structural_analysis.py
THIS_FILE = os.path.abspath(__file__)

# Project root: one level up from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))

# Add project root to sys.path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now we can safely import from src.*
from src.analysis.structural_paths import run_structural_paths

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "outputs", "site_level_synthetic.csv")


def main() -> None:
    print(f"Project root detected as: {PROJECT_ROOT}")
    print(f"Loading synthetic site-level data from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find synthetic data at:\n  {DATA_PATH}\n"
            "Make sure you have generated it first, e.g. by running:\n"
            "  python src/data_generation/generate_synthetic_sites_realistic.py"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n--- Data head ---")
    print(df.head())

    print("\nRunning structural path analysis...")
    run_structural_paths(df)


if __name__ == "__main__":
    main()
