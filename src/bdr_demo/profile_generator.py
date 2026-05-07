"""Synthetic 1RC profile generation for v0.1.

This module generates truth-layer battery profiles using the v0.1 1RC ECM:

    V(t)     = OCV(SOC(t)) - R0 * I(t) - V_RC(t)
    dV_RC/dt = -V_RC / (R1 * C1) + I(t) / C1

Sign convention:
    Positive current denotes discharge.

Scope:
    - truth-layer ProfileRow generation only
    - no sensor bias application
    - no module aggregation
    - no PyBaMM runtime dependency
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bdr_demo.schema import (
    SCHEMA_VERSION,
    CaseSpec,
    ProfileRow,
    rows_to_dataframe,
    validate_profile_row,
)


DEFAULT_OCV_TABLE_PATH = Path("data/ocv/ocv_chen2020.csv")


def load_ocv_table(path: str | Path = DEFAULT_OCV_TABLE_PATH) -> pd.DataFrame:
    """Load and validate a static OCV-SOC table.

    Required columns:
        - soc
        - ocv_V
    """

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"OCV table not found: {table_path}")

    df = pd.read_csv(table_path)

    required = {"soc", "ocv_V"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OCV table missing required columns: {missing}")

    df = df.sort_values("soc").reset_index(drop=True)

    if df["soc"].isna().any() or df["ocv_V"].isna().any():
        raise ValueError("OCV table contains NaN values in soc or ocv_V")

    if not df["soc"].is_monotonic_increasing:
        raise ValueError("OCV table SOC column must be monotonic increasing")

    if df["soc"].min() > 0.0 or df["soc"].max() < 1.0:
        raise ValueError("OCV table must cover SOC range [0, 1]")

    if not ((df["ocv_V"] > 2.0) & (df["ocv_V"] < 5.0)).all():
        raise ValueError("OCV table contains voltages outside plausible bounds")

    return df


def interpolate_ocv(soc: float | np.ndarray, ocv_table: pd.DataFrame) -> float | np.ndarray:
    """Interpolate OCV at one or multiple SOC values.

    SOC values are clipped to [0, 1] for interpolation.
    """

    soc_clipped = np.clip(soc, 0.0, 1.0)
    return np.interp(
        soc_clipped,
        ocv_table["soc"].to_numpy(dtype=float),
        ocv_table["ocv_V"].to_numpy(dtype=float),
    )


def make_time_grid(t_end_s: float, dt_s: float) -> np.ndarray:
    """Create an inclusive simulation time grid."""

    if t_end_s <= 0:
        raise ValueError("t_end_s must be positive")

    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    n_steps = int(round(t_end_s / dt_s))
    if abs(n_steps * dt_s - t_end_s) > 1e-9:
        raise ValueError("t_end_s must be an integer multiple of dt_s")

    return np.linspace(0.0, t_end_s, n_steps + 1)


def _current_array_from_constant(time_s: np.ndarray, current_A: float) -> np.ndarray:
    """Return a constant-current array."""

    return np.full_like(time_s, fill_value=float(current_A), dtype=float)


def generate_1rc_profile(
    case: CaseSpec,
    *,
    cell_id: int = 1,
    initial_soc_assumed: float = 0.80,
    discharge_current_A: float = 1.0,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
    ocv_table_path: str | Path = DEFAULT_OCV_TABLE_PATH,
) -> list[ProfileRow]:
    """Generate a truth-layer 1RC profile for one virtual cell.

    Args:
        case:
            Locked v0.1 perturbation case.
        cell_id:
            1-based virtual cell identifier.
        initial_soc_assumed:
            Diagnostic baseline SOC before applying physical inventory mismatch.
        discharge_current_A:
            Constant discharge current. Positive current denotes discharge.
        t_end_s:
            End time in seconds.
        dt_s:
            Time step in seconds.
        ocv_table_path:
            Static OCV-SOC CSV path.

    Returns:
        List of validated :class:`ProfileRow` objects.

    Notes:
        ``case.initial_soc_mismatch`` is applied to the true initial SOC:

            SOC_true(0) = initial_soc_assumed + initial_soc_mismatch

        This is a physical inventory mismatch, not an estimator output.
    """

    if cell_id < 1:
        raise ValueError("cell_id must be >= 1")

    if not 0.0 <= initial_soc_assumed <= 1.0:
        raise ValueError("initial_soc_assumed must be within [0, 1]")

    if discharge_current_A < 0:
        raise ValueError("discharge_current_A must be non-negative in v0.1")

    ocv_table = load_ocv_table(ocv_table_path)
    time_s = make_time_grid(t_end_s=t_end_s, dt_s=dt_s)
    current_A = _current_array_from_constant(time_s, discharge_current_A)

    capacity_Ah = case.nominal_capacity_Ah * case.capacity_factor
    r0_Ohm = case.r0_baseline_Ohm * case.contact_resistance_factor
    r1_Ohm = case.r1_baseline_Ohm
    c1_F = case.c1_baseline_F
    tau_s = r1_Ohm * c1_F

    if tau_s <= 0:
        raise ValueError("R1*C1 time constant must be positive")

    soc = float(np.clip(initial_soc_assumed + case.initial_soc_mismatch, 0.0, 1.0))
    v_rc = 0.0

    rows: list[ProfileRow] = []

    for idx, t_s in enumerate(time_s):
        i_A = float(current_A[idx])
        ocv_V = float(interpolate_ocv(soc, ocv_table))
        voltage_true_V = float(ocv_V - r0_Ohm * i_A - v_rc)

        row = ProfileRow(
            schema_version=SCHEMA_VERSION,
            case_id=case.case_id,
            level=case.level,
            cell_id=cell_id,
            t_s=float(t_s),
            current_A=i_A,
            voltage_true_V=voltage_true_V,
            ocv_V=ocv_V,
            soc_true=soc,
            v_rc_V=v_rc,
            capacity_Ah=capacity_Ah,
            capacity_factor=case.capacity_factor,
            contact_resistance_factor=case.contact_resistance_factor,
            r0_Ohm=r0_Ohm,
            r1_Ohm=r1_Ohm,
            c1_F=c1_F,
            fault_label=case.fault_label,
        )
        validate_profile_row(row)
        rows.append(row)

        # Explicit Euler update for next time step.
        # Positive current denotes discharge, therefore SOC decreases.
        if idx < len(time_s) - 1:
            dt = float(time_s[idx + 1] - time_s[idx])
            v_rc += dt * (-v_rc / tau_s + i_A / c1_F)
            soc -= i_A * dt / (3600.0 * capacity_Ah)
            soc = float(np.clip(soc, 0.0, 1.0))

    return rows


def profiles_to_dataframe(rows: Sequence[ProfileRow]) -> pd.DataFrame:
    """Convert profile rows to a DataFrame."""

    return rows_to_dataframe(rows)


__all__ = [
    "DEFAULT_OCV_TABLE_PATH",
    "load_ocv_table",
    "interpolate_ocv",
    "make_time_grid",
    "generate_1rc_profile",
    "profiles_to_dataframe",
]
