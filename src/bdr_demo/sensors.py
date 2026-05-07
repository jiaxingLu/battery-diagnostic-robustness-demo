"""Sensor-layer transformations for v0.1.

This module converts truth-layer ProfileRow objects into sensor-layer
SensedProfileRow objects.

v0.1 scope:
    - deterministic voltage bias only
    - no current sensor bias
    - no temperature sensor
    - no estimator output
    - no reported_soc

This preserves the signal-to-state separation:

    ProfileRow       = physical truth layer
    SensedProfileRow = measured signal layer
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from bdr_demo.schema import (
    CaseSpec,
    ProfileRow,
    SCHEMA_VERSION,
    SensedProfileRow,
    rows_to_dataframe,
    validate_sensed_profile_row,
)


def apply_voltage_sensor_bias(
    profile_rows: Sequence[ProfileRow],
    *,
    voltage_bias_mV: float,
) -> list[SensedProfileRow]:
    """Apply deterministic voltage sensor bias to truth-layer profile rows.

    Args:
        profile_rows:
            Truth-layer profile rows.
        voltage_bias_mV:
            Voltage sensor bias in millivolts. Positive values raise the
            measured voltage relative to the true terminal voltage.

    Returns:
        List of validated SensedProfileRow objects.
    """

    if not profile_rows:
        raise ValueError("profile_rows must not be empty")

    sensed_rows: list[SensedProfileRow] = []

    for row in profile_rows:
        voltage_measured_V = row.voltage_true_V + voltage_bias_mV / 1000.0

        sensed = SensedProfileRow(
            schema_version=SCHEMA_VERSION,
            case_id=row.case_id,
            level=row.level,
            cell_id=row.cell_id,
            t_s=row.t_s,
            current_A=row.current_A,
            voltage_true_V=row.voltage_true_V,
            voltage_measured_V=voltage_measured_V,
            voltage_bias_mV=voltage_bias_mV,
            ocv_V=row.ocv_V,
            soc_true=row.soc_true,
            v_rc_V=row.v_rc_V,
            capacity_Ah=row.capacity_Ah,
            capacity_factor=row.capacity_factor,
            contact_resistance_factor=row.contact_resistance_factor,
            r0_Ohm=row.r0_Ohm,
            r1_Ohm=row.r1_Ohm,
            c1_F=row.c1_F,
            fault_label=row.fault_label,
        )
        validate_sensed_profile_row(sensed)
        sensed_rows.append(sensed)

    return sensed_rows


def apply_case_sensor_model(
    case: CaseSpec,
    profile_rows: Sequence[ProfileRow],
) -> list[SensedProfileRow]:
    """Apply the v0.1 sensor model specified by a case.

    Currently this only applies ``case.voltage_bias_mV``.
    """

    return apply_voltage_sensor_bias(
        profile_rows,
        voltage_bias_mV=case.voltage_bias_mV,
    )


def sensed_profiles_to_dataframe(rows: Sequence[SensedProfileRow]) -> pd.DataFrame:
    """Convert sensed profile rows to a DataFrame."""

    return rows_to_dataframe(rows)


__all__ = [
    "apply_voltage_sensor_bias",
    "apply_case_sensor_model",
    "sensed_profiles_to_dataframe",
]
