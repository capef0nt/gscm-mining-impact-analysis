"""
Generate realistic synthetic site-level data using a simple
SEM-inspired generative process (no SDV).

Output:
    data/outputs/site_level_synthetic.csv  (synthetic sites with
    constructs, KPIs, OE_HARD, SAFETY_PERF)

This does NOT try to learn from the tiny real dataset. Instead it
encodes plausible mining behaviour:

- GSCM constructs (GPUR, GOPS, GLOG, GTRN, GCOL) share a common
  "green maturity" factor but are not identical.
- Mediators (SUPINT, MAINT, COMP) depend on relevant GSCM constructs.
- Perceived OE and EP depend on mediators.
- Objective KPIs (uptime, downtime, tons_per_hour, cost_per_ton, etc.)
  depend mainly on MAINT, COMP, SUPINT, and OE.
- Safety KPIs (ltifr, trifr, sifr, fifr) depend on MAINT, COMP, GTRN.
- OE_HARD and SAFETY_PERF are composite indices of the KPI layer.
"""

from __future__ import annotations

import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "outputs", "site_level_synthetic.csv")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _likert_from_z(z: np.ndarray, mean: float = 3.5, scale: float = 0.7) -> np.ndarray:
    """
    Map a latent z-score to a 1–5 Likert-style score.
    """
    x = mean + scale * z
    return np.clip(x, 1.0, 5.0)


def _standardize(x: pd.Series) -> pd.Series:
    """Standardize a series to z-scores."""
    return (x - x.mean()) / x.std(ddof=0)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------


def generate_synthetic_sites(
    n_sites: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic site-level dataset with:

    - Constructs: GPUR, GOPS, GLOG, GTRN, GCOL, SUPINT, MAINT, COMP, OE, EP
    - Operational KPIs
    - Safety KPIs
    - Composite indices OE_HARD, SAFETY_PERF
    """
    rng = np.random.default_rng(random_seed)

    # ---------------------------
    # 1) Latent drivers
    # ---------------------------
    # General "green maturity" factor
    z_gscm = rng.normal(0.0, 1.0, size=n_sites)
    # "pressure" factor (regulation + customers)
    z_pressure = rng.normal(0.0, 1.0, size=n_sites)
    # management support factor
    z_mgmt = rng.normal(0.0, 1.0, size=n_sites)
    # safety culture factor
    z_safety_culture = rng.normal(0.0, 1.0, size=n_sites)

    # ---------------------------
    # 2) GSCM constructs
    # ---------------------------
    # Use slightly different loadings + noise per construct so they’re
    # correlated but not identical.
    GPUR_z = 0.6 * z_gscm + 0.3 * z_pressure + rng.normal(0, 0.4, n_sites)
    GOPS_z = 0.7 * z_gscm + 0.2 * z_mgmt + rng.normal(0, 0.4, n_sites)
    GLOG_z = 0.6 * z_gscm + 0.3 * z_pressure + rng.normal(0, 0.5, n_sites)
    GTRN_z = 0.5 * z_gscm + 0.3 * z_mgmt + 0.3 * z_safety_culture + rng.normal(0, 0.4, n_sites)
    GCOL_z = 0.6 * z_gscm + 0.3 * z_pressure + rng.normal(0, 0.4, n_sites)

    GPUR = _likert_from_z(GPUR_z)
    GOPS = _likert_from_z(GOPS_z)
    GLOG = _likert_from_z(GLOG_z)
    GTRN = _likert_from_z(GTRN_z)
    GCOL = _likert_from_z(GCOL_z)

    # ---------------------------
    # 3) Mediators
    # ---------------------------
    SUPINT_z = 0.5 * GPUR_z + 0.3 * GCOL_z + rng.normal(0, 0.4, n_sites)
    MAINT_z = 0.5 * GOPS_z + 0.3 * GTRN_z + 0.2 * z_mgmt + rng.normal(0, 0.4, n_sites)
    COMP_z = 0.6 * GTRN_z + 0.2 * z_mgmt + rng.normal(0, 0.4, n_sites)

    SUPINT = _likert_from_z(SUPINT_z)
    MAINT = _likert_from_z(MAINT_z)
    COMP = _likert_from_z(COMP_z)

    # ---------------------------
    # 4) Perceived outcomes
    # ---------------------------
    OE_latent = (
        0.5 * MAINT_z
        + 0.3 * COMP_z
        + 0.2 * SUPINT_z
        + rng.normal(0, 0.3, n_sites)
    )
    EP_latent = (
        0.6 * OE_latent
        + 0.2 * z_pressure
        + rng.normal(0, 0.3, n_sites)
    )

    OE = _likert_from_z(OE_latent)
    EP = _likert_from_z(EP_latent)

    # ---------------------------
    # 5) Safety latent factor
    # ---------------------------
    safety_latent = (
        0.4 * MAINT_z
        + 0.4 * COMP_z
        + 0.2 * GTRN_z
        + 0.3 * z_safety_culture
        + rng.normal(0, 0.3, n_sites)
    )

    # ---------------------------
    # 6) Objective KPIs
    # ---------------------------
    # Normalize OE_latent and safety_latent for easier scaling
    OE_norm = (OE_latent - OE_latent.mean()) / OE_latent.std(ddof=0)
    SAF_norm = (safety_latent - safety_latent.mean()) / safety_latent.std(ddof=0)

    # Uptime: higher with OE
    uptime_percent = 85 + 7 * OE_norm + rng.normal(0, 2.0, n_sites)
    uptime_percent = np.clip(uptime_percent, 70, 99)

    # Unplanned downtime hours: lower with OE
    unplanned_downtime_hours = 180 - 35 * OE_norm + rng.normal(0, 10.0, n_sites)
    unplanned_downtime_hours = np.clip(unplanned_downtime_hours, 40, 260)

    # Tons per hour: higher with OE
    tons_per_hour = 260 + 45 * OE_norm + rng.normal(0, 15.0, n_sites)
    tons_per_hour = np.clip(tons_per_hour, 150, 420)

    # Rework rate: lower with OE and MAINT
    rework_rate_percent = 3.0 - 0.5 * OE_norm - 0.4 * _standardize(pd.Series(MAINT_z)) \
        + rng.normal(0, 0.3, n_sites)
    rework_rate_percent = np.clip(rework_rate_percent, 0.2, 6.0)

    # Energy & water per ton: lower with better GSCM and OE
    green_factor = _standardize(pd.Series(GPUR_z + GOPS_z + GLOG_z))
    energy_kwh_per_ton = 52 - 3.0 * green_factor - 1.5 * OE_norm + rng.normal(0, 1.5, n_sites)
    energy_kwh_per_ton = np.clip(energy_kwh_per_ton, 38, 60)

    water_m3_per_ton = 1.0 - 0.12 * green_factor - 0.08 * OE_norm + rng.normal(0, 0.05, n_sites)
    water_m3_per_ton = np.clip(water_m3_per_ton, 0.4, 1.5)

    # Cost structure: lower costs with OE and SUPINT + MAINT
    supply_maint_factor = _standardize(pd.Series(SUPINT_z + MAINT_z))
    cost_per_ton = 640 - 45 * OE_norm - 20 * supply_maint_factor + rng.normal(0, 20.0, n_sites)
    cost_per_ton = np.clip(cost_per_ton, 450, 800)

    maintenance_cost_per_ton = 160 - 25 * supply_maint_factor + rng.normal(0, 10.0, n_sites)
    maintenance_cost_per_ton = np.clip(maintenance_cost_per_ton, 80, 260)

    # Supply chain quality
    on_time_delivery_percent = 85 + 6 * supply_maint_factor + rng.normal(0, 3.0, n_sites)
    on_time_delivery_percent = np.clip(on_time_delivery_percent, 60, 99)

    supplier_defect_percent = 3.0 - 0.6 * green_factor - 0.4 * supply_maint_factor + rng.normal(0, 0.4, n_sites)
    supplier_defect_percent = np.clip(supplier_defect_percent, 0.1, 8.0)

    # ---------------------------
    # 7) Safety KPIs
    # ---------------------------
    # Base safety frequency ~ 0.5, improved with SAF_norm
    ltifr = 0.6 - 0.12 * SAF_norm + rng.normal(0, 0.05, n_sites)
    ltifr = np.clip(ltifr, 0.05, 1.2)

    trifr = 1.0 - 0.18 * SAF_norm + rng.normal(0, 0.08, n_sites)
    trifr = np.clip(trifr, 0.1, 2.0)

    sifr = 0.4 - 0.10 * SAF_norm + rng.normal(0, 0.04, n_sites)
    sifr = np.clip(sifr, 0.02, 0.9)

    fifr = 0.08 - 0.03 * SAF_norm + rng.normal(0, 0.02, n_sites)
    fifr = np.clip(fifr, 0.0, 0.2)

    safety_audits_passed_percent = 80 + 8 * SAF_norm + rng.normal(0, 4.0, n_sites)
    safety_audits_passed_percent = np.clip(safety_audits_passed_percent, 50, 100)

    employees_competent_percent = 75 + 7 * SAF_norm + rng.normal(0, 4.0, n_sites)
    employees_competent_percent = np.clip(employees_competent_percent, 50, 100)

    frontline_stoppages_percent = 30 - 5 * SAF_norm + rng.normal(0, 3.0, n_sites)
    frontline_stoppages_percent = np.clip(frontline_stoppages_percent, 5, 60)

    # ---------------------------
    # 8) OE_HARD and SAFETY_PERF indices
    # ---------------------------
    df_kpi = pd.DataFrame({
        "uptime_percent": uptime_percent,
        "unplanned_downtime_hours": unplanned_downtime_hours,
        "tons_per_hour": tons_per_hour,
        "rework_rate_percent": rework_rate_percent,
        "energy_kwh_per_ton": energy_kwh_per_ton,
        "water_m3_per_ton": water_m3_per_ton,
        "cost_per_ton": cost_per_ton,
        "maintenance_cost_per_ton": maintenance_cost_per_ton,
        "on_time_delivery_percent": on_time_delivery_percent,
        "supplier_defect_percent": supplier_defect_percent,
        "ltifr": ltifr,
        "trifr": trifr,
        "sifr": sifr,
        "fifr": fifr,
        "safety_audits_passed_percent": safety_audits_passed_percent,
        "employees_competent_percent": employees_competent_percent,
        "frontline_stoppages_percent": frontline_stoppages_percent,
    })

    # Operational hard index: high uptime, tons, on-time; low downtime, rework, costs, energy, water
    oe_components = pd.DataFrame({
        "uptime": _standardize(df_kpi["uptime_percent"]),
        "tons": _standardize(df_kpi["tons_per_hour"]),
        "on_time": _standardize(df_kpi["on_time_delivery_percent"]),
        "downtime": -_standardize(df_kpi["unplanned_downtime_hours"]),
        "rework": -_standardize(df_kpi["rework_rate_percent"]),
        "energy": -_standardize(df_kpi["energy_kwh_per_ton"]),
        "water": -_standardize(df_kpi["water_m3_per_ton"]),
        "cost": -_standardize(df_kpi["cost_per_ton"]),
        "maint_cost": -_standardize(df_kpi["maintenance_cost_per_ton"]),
    })
    OE_HARD = oe_components.mean(axis=1)

    # Safety performance index:
    # high audits & competence, low LTIFR/TRIFR/SIFR/FIFR, low stoppages, low supplier_defect
    safety_components = pd.DataFrame({
        "audits": _standardize(df_kpi["safety_audits_passed_percent"]),
        "competence": _standardize(df_kpi["employees_competent_percent"]),
        "ltifr": -_standardize(df_kpi["ltifr"]),
        "trifr": -_standardize(df_kpi["trifr"]),
        "sifr": -_standardize(df_kpi["sifr"]),
        "fifr": -_standardize(df_kpi["fifr"]),
        "stoppages": -_standardize(df_kpi["frontline_stoppages_percent"]),
        "sup_defects": -_standardize(df_kpi["supplier_defect_percent"]),
    })
    SAFETY_PERF = safety_components.mean(axis=1)

    # ---------------------------
    # 9) Assemble final DataFrame
    # ---------------------------
    df_constructs = pd.DataFrame({
        "site_id": [f"SYN_{i+1:03d}" for i in range(n_sites)],
        "GPUR": GPUR,
        "GOPS": GOPS,
        "GLOG": GLOG,
        "GTRN": GTRN,
        "GCOL": GCOL,
        "SUPINT": SUPINT,
        "MAINT": MAINT,
        "COMP": COMP,
        "OE": OE,
        "EP": EP,
    })

    df_all = pd.concat([df_constructs, df_kpi], axis=1)
    df_all["OE_HARD"] = OE_HARD
    df_all["SAFETY_PERF"] = SAFETY_PERF

    return df_all


def main(n_samples: int = 100, random_seed: int = 42) -> None:
    print(f"Generating {n_samples} synthetic sites (realistic model)...")
    df = generate_synthetic_sites(n_sites=n_samples, random_seed=random_seed)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Synthetic dataset saved to: {OUTPUT_PATH}")
    print("\n--- Synthetic Head ---")
    print(df.head())


if __name__ == "__main__":
    main(n_samples=100, random_seed=42)
