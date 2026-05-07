"""v0.1 diagnostic perturbation case matrix.

This module defines the eight controlled perturbation cases locked in
docs/methodology_decisions.md.

Important interpretation:
    ``level`` describes the design level of the injected perturbation.
    Cell-level cases represent uniform single-cell parameter deviations.
    Module-level cases represent explicit cell-to-cell scatter.

Downstream modules may lift uniform cell-level cases into a 24s1p series
module by replicating the same perturbed cell across all 24 positions.
"""

from __future__ import annotations

from typing import Iterable

from bdr_demo.schema import CaseSpec, validate_case_spec


N_SERIES_CELLS = 24


V0_1_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="baseline",
        level="cell",
        capacity_factor=1.00,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=0.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="none",
        notes="Healthy reference case.",
    ),
    CaseSpec(
        case_id="capacity_fade_90",
        level="cell",
        capacity_factor=0.90,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=0.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="capacity_fade",
        notes="Uniform 10% effective capacity reduction.",
    ),
    CaseSpec(
        case_id="contact_resistance_150",
        level="cell",
        capacity_factor=1.00,
        contact_resistance_factor=1.5,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=0.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="contact_resistance_growth",
        notes="Uniform 50% contact-resistance surrogate increase.",
    ),
    CaseSpec(
        case_id="initial_soc_mismatch_p05",
        level="cell",
        capacity_factor=1.00,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.05,
        voltage_bias_mV=0.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="initial_soc_inventory_offset",
        notes="True initial SOC exceeds BMS-assumed initial SOC by 5 percentage points.",
    ),
    CaseSpec(
        case_id="voltage_bias_p10mV",
        level="cell",
        capacity_factor=1.00,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=10.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="voltage_sensor_bias",
        notes="Uniform +10 mV voltage sensor bias; intentionally below 30 mV warning threshold.",
    ),
    CaseSpec(
        case_id="module_soc_scatter",
        level="module",
        capacity_factor=1.00,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=0.0,
        cell_scatter_type="soc",
        cell_scatter_magnitude=0.05,
        fault_label="cell_imbalance",
        notes="24s1p module with ±5% initial SOC scatter.",
    ),
    CaseSpec(
        case_id="module_capacity_scatter",
        level="module",
        capacity_factor=1.00,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.00,
        voltage_bias_mV=0.0,
        cell_scatter_type="capacity",
        cell_scatter_magnitude=0.05,
        fault_label="capacity_inconsistency",
        notes="24s1p module with ±5% cell capacity scatter around nominal capacity.",
    ),
    CaseSpec(
        case_id="combined_fault",
        level="module",
        capacity_factor=0.90,
        contact_resistance_factor=1.5,
        initial_soc_mismatch=0.05,
        voltage_bias_mV=10.0,
        cell_scatter_type="mixed",
        cell_scatter_magnitude=0.05,
        fault_label="combined",
        notes=(
            "Combined module-level case: capacity fade, contact-resistance growth, "
            "initial SOC mismatch, voltage sensor bias, and ±5% scatter."
        ),
    ),
)


def iter_cases() -> Iterable[CaseSpec]:
    """Return an iterator over the locked v0.1 case matrix."""

    return iter(V0_1_CASES)


def get_case(case_id: str) -> CaseSpec:
    """Return a case by ID.

    Raises:
        KeyError: if no case with the requested ID exists.
    """

    for case in V0_1_CASES:
        if case.case_id == case_id:
            return case

    raise KeyError(f"Unknown case_id: {case_id}")


def validate_case_matrix(cases: Iterable[CaseSpec] = V0_1_CASES) -> None:
    """Validate the full v0.1 case matrix."""

    case_list = list(cases)

    if len(case_list) != 8:
        raise ValueError(f"v0.1 case matrix must contain 8 cases, got {len(case_list)}")

    case_ids = [case.case_id for case in case_list]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"case_id values must be unique, got: {case_ids}")

    for case in case_list:
        validate_case_spec(case)

    required_ids = {
        "baseline",
        "capacity_fade_90",
        "contact_resistance_150",
        "initial_soc_mismatch_p05",
        "voltage_bias_p10mV",
        "module_soc_scatter",
        "module_capacity_scatter",
        "combined_fault",
    }
    missing = required_ids - set(case_ids)
    extra = set(case_ids) - required_ids
    if missing or extra:
        raise ValueError(f"case_id set mismatch: missing={missing}, extra={extra}")


# Validate immediately at import time. This is intentional: an invalid
# case matrix should fail early before any profile generation begins.
validate_case_matrix()


__all__ = [
    "N_SERIES_CELLS",
    "V0_1_CASES",
    "iter_cases",
    "get_case",
    "validate_case_matrix",
]
