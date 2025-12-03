from __future__ import annotations

import os
import sys
import pandas as pd

# Path bootstrapping
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.outer_model import compute_outer_model

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "outputs", "survey_synthetic.csv")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading synthetic survey data from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find synthetic survey at:\n  {DATA_PATH}\n"
            "Generate it first with:\n"
            "  python src/data_generation/generate_synthetic_survey.py"
        )

    df = pd.read_csv(DATA_PATH)
    print("\n--- Survey head ---")
    print(df.head())

    stats_df = compute_outer_model(df)

    print("\n--- Outer model stats ---")
    print(stats_df)


if __name__ == "__main__":
    main()
