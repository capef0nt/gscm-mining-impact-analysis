"""
Model configuration for gscm-mining-impact-analysis.

This file defines:
- Latent constructs and their survey indicators
- Structural model paths (PLS-SEM)
- Core KPI list for version 1
- Small helper functions to query the config

The goal is to keep all SEM/ML "wiring" in one place so that:
- docs/model_spec.md matches this file
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Likert scale configuration (for survey items)
# ---------------------------------------------------------------------------

LIKERT_MIN = 1
LIKERT_MAX = 5


# ---------------------------------------------------------------------------
# Construct configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstructConfig:
    """Configuration for a latent construct."""
    code: str
    name: str
    description: str
    indicators: List[str]


CONSTRUCTS: Dict[str, ConstructConfig] = {
    # GSCM dimensions --------------------------------------------------------
    "GPUR": ConstructConfig(
        code="GPUR",
        name="Green Purchasing",
        description=(
            "Extent to which the company considers environmental criteria "
            "in supplier selection, monitoring and certification."
        ),
        indicators=["GPUR_1", "GPUR_2", "GPUR_3", "GPUR_4"],
    ),
    "GOPS": ConstructConfig(
        code="GOPS",
        name="Green Operations",
        description=(
            "Degree to which production and processing operations are "
            "environmentally responsible and resource-efficient."
        ),
        indicators=["GOPS_1", "GOPS_2", "GOPS_3", "GOPS_4"],
    ),
    "GLOG": ConstructConfig(
        code="GLOG",
        name="Green Logistics",
        description=(
            "How green and efficient logistics and transport activities are."
        ),
        indicators=["GLOG_1", "GLOG_2", "GLOG_3"],
    ),
    "GTRN": ConstructConfig(
        code="GTRN",
        name="Green Training & Awareness",
        description=(
            "Quality and frequency of environmental and safety-related "
            "training and awareness."
        ),
        indicators=["GTRN_1", "GTRN_2", "GTRN_3"],
    ),
    "GCOL": ConstructConfig(
        code="GCOL",
        name="Green Collaboration",
        description=(
            "Extent of collaboration with suppliers and customers on "
            "environmental improvement initiatives."
        ),
        indicators=["GCOL_1", "GCOL_2", "GCOL_3"],
    ),

    # Mediators --------------------------------------------------------------
    "SUPINT": ConstructConfig(
        code="SUPINT",
        name="Supplier Integration",
        description=(
            "Degree of integration and information sharing with key suppliers."
        ),
        indicators=["SUPINT_1", "SUPINT_2", "SUPINT_3"],
    ),
    "MAINT": ConstructConfig(
        code="MAINT",
        name="Maintenance Quality",
        description=(
            "Robustness and effectiveness of the maintenance system."
        ),
        indicators=["MAINT_1", "MAINT_2", "MAINT_3"],
    ),
    "COMP": ConstructConfig(
        code="COMP",
        name="Employee Competence",
        description=(
            "Skill level, training and behavioural reliability of employees "
            "in operations."
        ),
        indicators=["COMP_1", "COMP_2", "COMP_3"],
    ),

    # Perceived outcomes -----------------------------------------------------
    "OE": ConstructConfig(
        code="OE",
        name="Perceived Operational Efficiency",
        description=(
            "Subjective perception of how efficiently operations run "
            "(downtime, resource use, smoothness, cost)."
        ),
        indicators=["OE_1", "OE_2", "OE_3", "OE_4", "OE_5"],
    ),
    "EP": ConstructConfig(
        code="EP",
        name="Perceived Enterprise Performance",
        description=(
            "Subjective perception of overall business performance and "
            "competitiveness."
        ),
        indicators=["EP_1", "EP_2", "EP_3", "EP_4", "EP_5"],
    ),
}


def get_construct_codes() -> List[str]:
    """Return list of construct codes in a stable order."""
    # Order roughly from inputs → mediators → outcomes
    ordered = [
        "GPUR", "GOPS", "GLOG", "GTRN", "GCOL",
        "SUPINT", "MAINT", "COMP",
        "OE", "EP",
    ]
    # Fallback in case dict changes
    return [c for c in ordered if c in CONSTRUCTS] + [
        c for c in CONSTRUCTS.keys() if c not in ordered
    ]


def get_construct(code: str) -> ConstructConfig:
    """Get configuration for a single construct code."""
    return CONSTRUCTS[code]


def all_indicators() -> List[str]:
    """Return a flat list of all indicator column names."""
    cols: List[str] = []
    for cfg in CONSTRUCTS.values():
        cols.extend(cfg.indicators)
    return cols


# ---------------------------------------------------------------------------
# Structural model (PLS-SEM paths)
# ---------------------------------------------------------------------------

# Each path is a tuple (source_construct, target_construct).
# Example: ("GOPS", "MAINT") means GOPS → MAINT.

STRUCTURAL_PATHS: List[Tuple[str, str]] = [
    # GSCM → Mediators
    ("GPUR", "SUPINT"),
    ("GPUR", "MAINT"),

    ("GOPS", "MAINT"),
    ("GOPS", "COMP"),

    ("GLOG", "SUPINT"),

    ("GTRN", "COMP"),
    ("GTRN", "MAINT"),

    ("GCOL", "SUPINT"),
    ("GCOL", "GOPS"),

    # Mediators → OE
    ("SUPINT", "OE"),
    ("MAINT", "OE"),
    ("COMP", "OE"),

    # OE → EP
    ("OE", "EP"),
]


def get_downstream_targets(source: str) -> List[str]:
    """Return all constructs that the given construct points to."""
    return [t for s, t in STRUCTURAL_PATHS if s == source]


def get_upstream_sources(target: str) -> List[str]:
    """Return all constructs that point into the given construct."""
    return [s for s, t in STRUCTURAL_PATHS if t == target]


# ---------------------------------------------------------------------------
# Core KPI configuration (Version 1)
# ---------------------------------------------------------------------------

# Core KPIs that we expect to be available for a site in v1.
# These are the targets that ML models will (initially) focus on.

CORE_KPIS: List[str] = [
    # Operational
    "uptime_percent",
    "unplanned_downtime_hours",
    "tons_per_hour",
    "rework_rate_percent",

    # Environmental
    "energy_kwh_per_ton",
    "water_m3_per_ton",

    # Cost
    "cost_per_ton",
    "maintenance_cost_per_ton",

    # Supply chain
    "on_time_delivery_percent",
    "supplier_defect_percent",

    # Safety
    "ltifr",
    "trifr",
    "sifr",
    "fifr"
    "safety_audits_passed_percent",
    "employees_competent_percent",
]


# Optional: informal grouping for reporting / plotting
KPI_CATEGORIES: Dict[str, List[str]] = {
    "operational": [
        "uptime_percent",
        "unplanned_downtime_hours",
        "tons_per_hour",
        "rework_rate_percent",
    ],
    "environmental": [
        "energy_kwh_per_ton",
        "water_m3_per_ton",
    ],
    "cost": [
        "cost_per_ton",
        "maintenance_cost_per_ton",
    ],
    "supply_chain": [
        "on_time_delivery_percent",
        "supplier_defect_percent",
    ],
    "safety": [
        "ltifr",
        "trifr",
        "sifr",
        "fifr",
        "frontline_stoppages_percent",
        "near_miss_reports",
        "critical_control_compliance_percent",
        "iso_45001_certified",
    ]

}


def is_core_kpi(name: str) -> bool:
    """Check if a KPI name is part of the core v1 set."""
    return name in CORE_KPIS


def list_kpis_by_category(category: str) -> List[str]:
    """Return KPIs for a given category (operational, environmental, etc.)."""
    return KPI_CATEGORIES.get(category, [])


def list_all_kpis() -> List[str]:
    """Return a flat list of all KPIs known to the config."""
    seen = set()
    all_list: List[str] = []
    for group in KPI_CATEGORIES.values():
        for k in group:
            if k not in seen:
                seen.add(k)
                all_list.append(k)
    return all_list

