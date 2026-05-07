"""Schema validation tests for battery-diagnostic-robustness-demo v0.1.

This test module covers:

- happy-path validation for CaseSpec, ProfileRow, and SensedProfileRow;
- representative failure paths in validators;
- cross-field consistency rules for scatter definitions;
- frozen-dataclass enforcement;
- schema_version mismatch detection;
- smoke instantiation for all schema dataclasses;
- regression guard against reintroducing reported_soc into SensedProfileRow.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from bdr_demo.schema import (
    SCHEMA_VERSION,
    CaseSpec,
    DiagnosticReportRow,
    ModuleInconsistencyReportRow,
    ObservabilityRow,
    ProfileRow,
    SensedProfileRow,
    rows_to_dataframe,
    validate_case_spec,
    validate_profile_row,
    validate_sensed_profile_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline_case(**overrides) -> CaseSpec:
    defaults = dict(
        case_id="baseline",
        level="cell",
        capacity_factor=1.0,
        contact_resistance_factor=1.0,
        initial_soc_mismatch=0.0,
        voltage_bias_mV=0.0,
        cell_scatter_type="none",
        cell_scatter_magnitude=0.0,
        fault_label="none",
    )
    defaults.update(overrides)
    return CaseSpec(**defaults)


def _make_baseline_profile_row(**overrides) -> ProfileRow:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        case_id="baseline",
        level="cell",
        cell_id=1,
        t_s=0.0,
        current_A=1.0,
        voltage_true_V=3.7,
        ocv_V=3.75,
        soc_true=0.8,
        v_rc_V=0.0,
        capacity_Ah=3.5,
        capacity_factor=1.0,
        contact_resistance_factor=1.0,
        r0_Ohm=0.05,
        r1_Ohm=0.02,
        c1_F=500.0,
        fault_label="none",
    )
    defaults.update(overrides)
    return ProfileRow(**defaults)


def _make_baseline_sensed_row(**overrides) -> SensedProfileRow:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        case_id="baseline",
        level="cell",
        cell_id=1,
        t_s=0.0,
        current_A=1.0,
        voltage_true_V=3.7,
        voltage_measured_V=3.7,
        voltage_bias_mV=0.0,
        ocv_V=3.75,
        soc_true=0.8,
        v_rc_V=0.0,
        capacity_Ah=3.5,
        capacity_factor=1.0,
        contact_resistance_factor=1.0,
        r0_Ohm=0.05,
        r1_Ohm=0.02,
        c1_F=500.0,
        fault_label="none",
    )
    defaults.update(overrides)
    return SensedProfileRow(**defaults)


# ---------------------------------------------------------------------------
# CaseSpec validation
# ---------------------------------------------------------------------------


def test_valid_case_spec_passes():
    validate_case_spec(_make_baseline_case())


def test_case_spec_negative_capacity_factor_fails():
    with pytest.raises(ValueError, match="capacity_factor"):
        validate_case_spec(
            _make_baseline_case(case_id="bad_capacity", capacity_factor=-1.0)
        )


def test_case_spec_negative_r0_fails():
    with pytest.raises(ValueError, match="r0_baseline_Ohm"):
        validate_case_spec(
            _make_baseline_case(case_id="bad_r0", r0_baseline_Ohm=-0.05)
        )


def test_case_spec_cell_level_with_scatter_fails():
    with pytest.raises(ValueError, match="cell-level case"):
        validate_case_spec(
            _make_baseline_case(
                case_id="invalid_combo",
                level="cell",
                cell_scatter_type="soc",
                cell_scatter_magnitude=0.05,
                fault_label="cell_imbalance",
            )
        )


def test_case_spec_scatter_type_none_with_magnitude_fails():
    with pytest.raises(ValueError, match="cell_scatter_type='none'"):
        validate_case_spec(
            _make_baseline_case(
                case_id="inconsistent_scatter_a",
                level="module",
                cell_scatter_type="none",
                cell_scatter_magnitude=0.05,
            )
        )


def test_case_spec_scatter_type_nonzero_with_zero_magnitude_fails():
    with pytest.raises(ValueError, match="cell_scatter_type != 'none'"):
        validate_case_spec(
            _make_baseline_case(
                case_id="inconsistent_scatter_b",
                level="module",
                cell_scatter_type="capacity",
                cell_scatter_magnitude=0.0,
                fault_label="capacity_inconsistency",
            )
        )


def test_case_spec_invalid_fault_label_fails():
    with pytest.raises(ValueError, match="fault_label"):
        validate_case_spec(
            _make_baseline_case(
                case_id="bad_fault",
                fault_label="not_a_real_fault",  # type: ignore[arg-type]
            )
        )


def test_frozen_case_spec_cannot_be_modified():
    case = _make_baseline_case()
    with pytest.raises(FrozenInstanceError):
        case.case_id = "modified"  # type: ignore[misc]


def test_module_level_case_with_consistent_scatter_passes():
    validate_case_spec(
        _make_baseline_case(
            case_id="module_soc_scatter",
            level="module",
            cell_scatter_type="soc",
            cell_scatter_magnitude=0.05,
            fault_label="cell_imbalance",
        )
    )


# ---------------------------------------------------------------------------
# ProfileRow validation
# ---------------------------------------------------------------------------


def test_valid_profile_row_passes():
    validate_profile_row(_make_baseline_profile_row())


def test_profile_row_soc_out_of_range_fails():
    with pytest.raises(ValueError, match="soc_true"):
        validate_profile_row(_make_baseline_profile_row(soc_true=1.2))


def test_profile_row_invalid_schema_version_fails():
    with pytest.raises(ValueError, match="schema_version"):
        validate_profile_row(
            _make_baseline_profile_row(schema_version="wrong_version")
        )


def test_profile_row_invalid_fault_label_fails():
    with pytest.raises(ValueError, match="fault_label"):
        validate_profile_row(
            _make_baseline_profile_row(
                fault_label="not_a_real_fault"  # type: ignore[arg-type]
            )
        )


# ---------------------------------------------------------------------------
# SensedProfileRow validation
# ---------------------------------------------------------------------------


def test_valid_sensed_profile_row_passes():
    validate_sensed_profile_row(_make_baseline_sensed_row())


def test_sensed_profile_row_voltage_bias_consistency_passes():
    row = _make_baseline_sensed_row(
        voltage_true_V=3.7,
        voltage_bias_mV=10.0,
        voltage_measured_V=3.71,
    )
    validate_sensed_profile_row(row)


def test_sensed_profile_row_voltage_bias_mismatch_fails():
    row = _make_baseline_sensed_row(
        voltage_true_V=3.7,
        voltage_bias_mV=10.0,
        voltage_measured_V=3.70,
    )
    with pytest.raises(ValueError, match="voltage_measured_V"):
        validate_sensed_profile_row(row)


# ---------------------------------------------------------------------------
# Contract-regression guards
# ---------------------------------------------------------------------------


def test_sensed_profile_row_no_reported_soc():
    field_names = {field.name for field in fields(SensedProfileRow)}
    assert "reported_soc" not in field_names, (
        "SensedProfileRow must not contain reported_soc per v0.1 contract. "
        "BMS estimator output belongs to diagnostics/reporting modules."
    )


# ---------------------------------------------------------------------------
# Smoke instantiation for remaining dataclasses
# ---------------------------------------------------------------------------


def test_sensed_profile_row_instantiates():
    row = _make_baseline_sensed_row()
    assert row.case_id == "baseline"


def test_diagnostic_report_row_instantiates():
    row = DiagnosticReportRow(
        schema_version=SCHEMA_VERSION,
        case_id="baseline",
        fault_label="none",
        soc_error_coulomb_max=0.0,
        soc_error_ocv_reset_max=0.0,
        voltage_residual_max_mV=0.0,
        capacity_consistency_error=0.0,
        contact_R_inferred_ratio=1.0,
        module_delta_voltage_max_mV=0.0,
        coulomb_drift_verdict="PASS",
        ocv_reset_verdict="PASS",
        voltage_residual_verdict="PASS",
        capacity_verdict="PASS",
        contact_R_verdict="PASS",
        module_imbalance_verdict="PASS",
        false_positive=False,
        false_negative=False,
        final_verdict="PASS_HEALTHY",
    )
    assert row.final_verdict == "PASS_HEALTHY"


def test_observability_row_instantiates():
    row = ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="capacity_fade",
        voltage_signature="discharge curve shifted lower",
        current_signature="no signature",
        temperature_signature="no signature",
        soc_signature="OCV-derived SOC inconsistency under capacity shift",
        soh_signature="reduced effective capacity",
        module_spread_signature="no signature in uniform cell-level case",
        ambiguity_level="medium",
        notes=(
            "Distinguishable from contact resistance growth only when "
            "capacity-consistency information is available."
        ),
    )
    assert row.fault_label == "capacity_fade"


def test_module_inconsistency_report_row_instantiates():
    row = ModuleInconsistencyReportRow(
        schema_version=SCHEMA_VERSION,
        case_id="module_soc_scatter",
        pack_voltage_max_V=88.0,
        pack_voltage_min_V=72.0,
        cell_voltage_max_anytime_V=4.10,
        cell_voltage_min_anytime_V=3.05,
        delta_cell_voltage_max_mV=120.0,
        weakest_cell_id=7,
        weakest_cell_criterion="lowest_min_voltage",
        module_risk_flag="WARNING",
    )
    assert row.weakest_cell_id == 7
    assert row.weakest_cell_criterion == "lowest_min_voltage"


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def test_rows_to_dataframe_exports_expected_columns():
    df = rows_to_dataframe([_make_baseline_case()])
    expected_first_columns = [
        "case_id",
        "level",
        "capacity_factor",
        "contact_resistance_factor",
        "initial_soc_mismatch",
        "voltage_bias_mV",
        "cell_scatter_type",
        "cell_scatter_magnitude",
        "fault_label",
    ]
    assert list(df.columns)[: len(expected_first_columns)] == expected_first_columns
