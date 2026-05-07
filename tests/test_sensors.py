"""Tests for v0.1 sensor-layer transformations."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from bdr_demo.case_matrix import get_case
from bdr_demo.profile_generator import generate_1rc_profile
from bdr_demo.sensors import (
    apply_case_sensor_model,
    apply_voltage_sensor_bias,
    sensed_profiles_to_dataframe,
)
from bdr_demo.schema import SensedProfileRow, validate_sensed_profile_row


def _baseline_truth_rows():
    case = get_case("baseline")
    return generate_1rc_profile(
        case,
        cell_id=1,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=10.0,
        dt_s=1.0,
    )


def test_apply_zero_voltage_bias_preserves_voltage():
    truth_rows = _baseline_truth_rows()
    sensed_rows = apply_voltage_sensor_bias(truth_rows, voltage_bias_mV=0.0)

    assert len(sensed_rows) == len(truth_rows)

    for truth, sensed in zip(truth_rows, sensed_rows):
        assert sensed.voltage_measured_V == truth.voltage_true_V
        assert sensed.voltage_bias_mV == 0.0
        validate_sensed_profile_row(sensed)


def test_apply_positive_voltage_bias_shifts_measured_voltage_by_10mV():
    truth_rows = _baseline_truth_rows()
    sensed_rows = apply_voltage_sensor_bias(truth_rows, voltage_bias_mV=10.0)

    for truth, sensed in zip(truth_rows, sensed_rows):
        assert np.isclose(
            sensed.voltage_measured_V,
            truth.voltage_true_V + 0.010,
        )
        assert sensed.voltage_bias_mV == 10.0


def test_apply_negative_voltage_bias_shifts_measured_voltage_down():
    truth_rows = _baseline_truth_rows()
    sensed_rows = apply_voltage_sensor_bias(truth_rows, voltage_bias_mV=-5.0)

    for truth, sensed in zip(truth_rows, sensed_rows):
        assert np.isclose(
            sensed.voltage_measured_V,
            truth.voltage_true_V - 0.005,
        )


def test_apply_case_sensor_model_uses_case_voltage_bias():
    case = get_case("voltage_bias_p10mV")
    truth_rows = generate_1rc_profile(
        case,
        cell_id=1,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=10.0,
        dt_s=1.0,
    )

    sensed_rows = apply_case_sensor_model(case, truth_rows)

    assert all(row.voltage_bias_mV == 10.0 for row in sensed_rows)
    assert np.isclose(
        sensed_rows[0].voltage_measured_V,
        truth_rows[0].voltage_true_V + 0.010,
    )


def test_sensor_layer_preserves_truth_fields():
    truth_rows = _baseline_truth_rows()
    sensed_rows = apply_voltage_sensor_bias(truth_rows, voltage_bias_mV=10.0)

    for truth, sensed in zip(truth_rows, sensed_rows):
        assert sensed.schema_version == truth.schema_version
        assert sensed.case_id == truth.case_id
        assert sensed.level == truth.level
        assert sensed.cell_id == truth.cell_id
        assert sensed.t_s == truth.t_s
        assert sensed.current_A == truth.current_A
        assert sensed.voltage_true_V == truth.voltage_true_V
        assert sensed.ocv_V == truth.ocv_V
        assert sensed.soc_true == truth.soc_true
        assert sensed.v_rc_V == truth.v_rc_V
        assert sensed.capacity_Ah == truth.capacity_Ah
        assert sensed.fault_label == truth.fault_label


def test_sensed_profile_row_still_has_no_reported_soc():
    field_names = {field.name for field in fields(SensedProfileRow)}
    assert "reported_soc" not in field_names


def test_apply_voltage_sensor_bias_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        apply_voltage_sensor_bias([], voltage_bias_mV=0.0)


def test_sensed_profiles_to_dataframe_exports_expected_columns():
    truth_rows = _baseline_truth_rows()
    sensed_rows = apply_voltage_sensor_bias(truth_rows, voltage_bias_mV=10.0)
    df = sensed_profiles_to_dataframe(sensed_rows)

    expected = [
        "schema_version",
        "case_id",
        "level",
        "cell_id",
        "t_s",
        "current_A",
        "voltage_true_V",
        "voltage_measured_V",
        "voltage_bias_mV",
        "ocv_V",
        "soc_true",
    ]

    assert list(df.columns)[: len(expected)] == expected
    assert len(df) == len(truth_rows)
