"""24s1p pure-series module aggregation for v0.1.

This module lifts cell-level synthetic truth profiles into a virtual
24s1p pure-series module.

Scope:
    - pure series topology only
    - same prescribed current through all cells
    - no parallel current redistribution
    - no thermal coupling
    - no sensor layer
    - no diagnostics

The aggregation is intentionally simple:

    V_pack(t) = sum_i V_cell_i(t)

Cell-to-cell scatter is applied deterministically so that test outputs are
reproducible.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from bdr_demo.case_matrix import N_SERIES_CELLS
from bdr_demo.profile_generator import generate_1rc_profile
from bdr_demo.schema import (
    CaseSpec,
    ModuleInconsistencyReportRow,
    ProfileRow,
    RuleVerdict,
    SCHEMA_VERSION,
)


_CURRENT_TOL_A = 1e-9


def make_deterministic_scatter(n_cells: int, magnitude: float) -> np.ndarray:
    """Return deterministic centered scatter values.

    Args:
        n_cells:
            Number of cells.
        magnitude:
            Scatter half-width. Example: 0.05 yields values spanning
            [-0.05, +0.05].

    Returns:
        Array of length ``n_cells``.

    Notes:
        The values are deterministic, monotonic, and centered close to zero.
        This is deliberate: v0.1 tests should be reproducible without random
        seeds or stochastic assumptions.
    """

    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")

    if magnitude == 0:
        return np.zeros(n_cells, dtype=float)

    return np.linspace(-magnitude, magnitude, n_cells, dtype=float)


def _effective_cell_case(
    case: CaseSpec,
    *,
    cell_id: int,
    soc_scatter: float,
    capacity_scatter: float,
) -> CaseSpec:
    """Create a per-cell effective case after applying scatter.

    The returned case keeps the original ``case_id`` and ``fault_label`` but
    removes explicit scatter fields, because scatter has already been applied
    to the effective per-cell parameters.
    """

    capacity_factor = case.capacity_factor * (1.0 + capacity_scatter)
    if capacity_factor <= 0:
        raise ValueError(
            f"capacity scatter produced non-positive capacity factor for cell {cell_id}"
        )

    return replace(
        case,
        level=case.level,
        capacity_factor=capacity_factor,
        initial_soc_mismatch=case.initial_soc_mismatch + soc_scatter,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        notes=f"{case.notes} | effective cell_id={cell_id}",
    )


def generate_series_module_cell_profiles(
    case: CaseSpec,
    *,
    n_series_cells: int = N_SERIES_CELLS,
    initial_soc_assumed: float = 0.80,
    discharge_current_A: float = 1.0,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
) -> list[ProfileRow]:
    """Generate per-cell truth profiles for a virtual pure-series module.

    For cell-level uniform cases, the same effective cell is replicated across
    all 24 series positions.

    For module-level scatter cases, deterministic scatter is applied before
    profile generation:

    - ``cell_scatter_type == "soc"``: initial SOC mismatch scatter.
    - ``cell_scatter_type == "capacity"``: capacity factor scatter.
    - ``cell_scatter_type == "mixed"``: both SOC and capacity scatter.

    Contact resistance is not scattered in v0.1; it is applied uniformly via
    ``case.contact_resistance_factor``.
    """

    if n_series_cells <= 0:
        raise ValueError("n_series_cells must be positive")

    scatter = make_deterministic_scatter(n_series_cells, case.cell_scatter_magnitude)

    all_rows: list[ProfileRow] = []

    for idx in range(n_series_cells):
        cell_id = idx + 1

        soc_scatter = 0.0
        capacity_scatter = 0.0

        if case.cell_scatter_type in ("soc", "mixed"):
            soc_scatter = float(scatter[idx])

        if case.cell_scatter_type in ("capacity", "mixed"):
            capacity_scatter = float(scatter[idx])

        effective_case = _effective_cell_case(
            case,
            cell_id=cell_id,
            soc_scatter=soc_scatter,
            capacity_scatter=capacity_scatter,
        )

        rows = generate_1rc_profile(
            effective_case,
            cell_id=cell_id,
            initial_soc_assumed=initial_soc_assumed,
            discharge_current_A=discharge_current_A,
            t_end_s=t_end_s,
            dt_s=dt_s,
        )
        all_rows.extend(rows)

    return all_rows


def aggregate_series_module(cell_profiles: Sequence[ProfileRow]) -> pd.DataFrame:
    """Aggregate cell profiles into pack-level pure-series module signals.

    Returns:
        DataFrame with one row per time point and columns:

        - schema_version
        - case_id
        - t_s
        - current_A
        - pack_voltage_true_V
        - min_cell_voltage_V
        - max_cell_voltage_V
        - delta_cell_voltage_mV
        - n_cells

    Raises:
        ValueError if the input is empty, case IDs are mixed, or current is
        inconsistent across cells at the same time point.
    """

    if not cell_profiles:
        raise ValueError("cell_profiles must not be empty")

    df = pd.DataFrame([row.__dict__ for row in cell_profiles])

    if df["case_id"].nunique() != 1:
        raise ValueError("cell_profiles must contain exactly one case_id")

    grouped = df.groupby("t_s", sort=True)

    rows: list[dict[str, object]] = []
    for t_s, group in grouped:
        current_span = group["current_A"].max() - group["current_A"].min()
        if abs(current_span) > _CURRENT_TOL_A:
            raise ValueError(f"inconsistent current across cells at t_s={t_s}")

        min_v = float(group["voltage_true_V"].min())
        max_v = float(group["voltage_true_V"].max())

        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": str(group["case_id"].iloc[0]),
                "t_s": float(t_s),
                "current_A": float(group["current_A"].iloc[0]),
                "pack_voltage_true_V": float(group["voltage_true_V"].sum()),
                "min_cell_voltage_V": min_v,
                "max_cell_voltage_V": max_v,
                "delta_cell_voltage_mV": 1000.0 * (max_v - min_v),
                "n_cells": int(group["cell_id"].nunique()),
            }
        )

    module_df = pd.DataFrame(rows)

    if not module_df["t_s"].is_monotonic_increasing:
        raise RuntimeError("Aggregated module time axis is not monotonic increasing")

    return module_df


def make_module_inconsistency_report(
    cell_profiles: Sequence[ProfileRow],
    module_df: pd.DataFrame,
    *,
    weakest_cell_criterion: str = "lowest_min_voltage",
) -> ModuleInconsistencyReportRow:
    """Create a module inconsistency report row.

    Supported weakest-cell criteria:

    - ``lowest_min_voltage``
    - ``lowest_capacity``
    - ``highest_R0``
    """

    if not cell_profiles:
        raise ValueError("cell_profiles must not be empty")

    if module_df.empty:
        raise ValueError("module_df must not be empty")

    cell_df = pd.DataFrame([row.__dict__ for row in cell_profiles])
    case_id = str(cell_df["case_id"].iloc[0])

    if weakest_cell_criterion == "lowest_min_voltage":
        weakest_cell_id = int(cell_df.groupby("cell_id")["voltage_true_V"].min().idxmin())
    elif weakest_cell_criterion == "lowest_capacity":
        weakest_cell_id = int(cell_df.groupby("cell_id")["capacity_Ah"].first().idxmin())
    elif weakest_cell_criterion == "highest_R0":
        weakest_cell_id = int(cell_df.groupby("cell_id")["r0_Ohm"].first().idxmax())
    else:
        raise ValueError(f"Unknown weakest_cell_criterion: {weakest_cell_criterion}")

    delta_max_mV = float(module_df["delta_cell_voltage_mV"].max())
    risk: RuleVerdict
    if delta_max_mV > 50.0:
        risk = "FAIL"
    elif delta_max_mV > 30.0:
        risk = "WARNING"
    else:
        risk = "PASS"

    return ModuleInconsistencyReportRow(
        schema_version=SCHEMA_VERSION,
        case_id=case_id,
        pack_voltage_max_V=float(module_df["pack_voltage_true_V"].max()),
        pack_voltage_min_V=float(module_df["pack_voltage_true_V"].min()),
        cell_voltage_max_anytime_V=float(cell_df["voltage_true_V"].max()),
        cell_voltage_min_anytime_V=float(cell_df["voltage_true_V"].min()),
        delta_cell_voltage_max_mV=delta_max_mV,
        weakest_cell_id=weakest_cell_id,
        weakest_cell_criterion=weakest_cell_criterion,  # type: ignore[arg-type]
        module_risk_flag=risk,
    )


__all__ = [
    "make_deterministic_scatter",
    "generate_series_module_cell_profiles",
    "aggregate_series_module",
    "make_module_inconsistency_report",
]
