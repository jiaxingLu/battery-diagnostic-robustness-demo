"""Tests for v0.1 rule-based diagnostics."""

from __future__ import annotations

import numpy as np

from bdr_demo.case_matrix import get_case
from bdr_demo.diagnostics import (
    compute_contact_R_inferred_ratio,
    compute_diagnostic_report,
    compute_module_delta_voltage_max_mV,
    compute_voltage_residual_max_mV,
    combine_final_verdict,
    rule_verdict_from_abs_error,
    rule_verdict_from_upper_bound,
)
from bdr_demo.module_aggregator import (
    aggregate_series_module,
    generate_series_module_cell_profiles,
)
from bdr_demo.profile_generator import generate_1rc_profile
from bdr_demo.sensors import apply_case_sensor_model


def _sensed_single_cell(case_id: str):
    case = get_case(case_id)
    truth = generate_1rc_profile(
        case,
        cell_id=1,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )
    sensed = apply_case_sensor_model(case, truth)
    return case, sensed


def test_rule_verdict_from_abs_error_thresholds():
    assert rule_verdict_from_abs_error(
        0.01, warning_threshold=0.03, fail_threshold=0.05
    ) == "PASS"
    assert rule_verdict_from_abs_error(
        0.03, warning_threshold=0.03, fail_threshold=0.05
    ) == "WARNING"
    assert rule_verdict_from_abs_error(
        0.05, warning_threshold=0.03, fail_threshold=0.05
    ) == "FAIL"


def test_rule_verdict_from_upper_bound_thresholds():
    assert rule_verdict_from_upper_bound(
        10.0, warning_threshold=30.0, fail_threshold=100.0
    ) == "PASS"
    assert rule_verdict_from_upper_bound(
        40.0, warning_threshold=30.0, fail_threshold=100.0
    ) == "WARNING"
    assert rule_verdict_from_upper_bound(
        120.0, warning_threshold=30.0, fail_threshold=100.0
    ) == "FAIL"


def test_baseline_report_is_pass_healthy():
    case, sensed = _sensed_single_cell("baseline")
    report = compute_diagnostic_report(case, sensed)

    assert report.case_id == "baseline"
    assert report.fault_label == "none"
    assert report.false_positive is False
    assert report.false_negative is False
    assert report.final_verdict == "PASS_HEALTHY"


def test_capacity_fade_is_detected_by_capacity_or_coulomb_rule():
    case, sensed = _sensed_single_cell("capacity_fade_90")
    report = compute_diagnostic_report(case, sensed)

    assert report.capacity_consistency_error > 0.09
    assert report.capacity_verdict == "FAIL"
    assert report.final_verdict == "PASS_DETECTED"


def test_contact_resistance_growth_is_detected():
    case, sensed = _sensed_single_cell("contact_resistance_150")
    report = compute_diagnostic_report(case, sensed)

    assert np.isclose(compute_contact_R_inferred_ratio(case, sensed), 1.5)
    assert report.contact_R_verdict == "FAIL"
    assert report.final_verdict == "PASS_DETECTED"


def test_initial_soc_inventory_offset_is_detected_by_coulomb_drift():
    case, sensed = _sensed_single_cell("initial_soc_mismatch_p05")
    report = compute_diagnostic_report(case, sensed)

    assert report.soc_error_coulomb_max >= 0.05
    assert report.coulomb_drift_verdict == "FAIL"
    assert report.final_verdict == "PASS_DETECTED"


def test_voltage_bias_10mV_is_below_warning_and_marked_missed():
    case, sensed = _sensed_single_cell("voltage_bias_p10mV")
    report = compute_diagnostic_report(case, sensed)

    assert np.isclose(compute_voltage_residual_max_mV(sensed), 10.0)
    assert report.voltage_residual_verdict == "PASS"
    assert report.false_negative is True
    assert report.final_verdict == "FAIL_MISSED_FAULT"


def test_module_delta_voltage_defaults_to_zero_without_module_df():
    assert compute_module_delta_voltage_max_mV(None) == 0.0


def test_module_soc_scatter_is_detected_by_module_imbalance():
    case = get_case("module_soc_scatter")
    truth = generate_series_module_cell_profiles(
        case,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )
    module_df = aggregate_series_module(truth)
    sensed = apply_case_sensor_model(case, truth)

    report = compute_diagnostic_report(case, sensed, module_df=module_df)

    assert report.module_delta_voltage_max_mV > 30.0
    assert report.module_imbalance_verdict in {"WARNING", "FAIL"}
    assert report.final_verdict in {"WARNING_BOUNDARY", "PASS_DETECTED"}


def test_combined_fault_is_detected():
    case = get_case("combined_fault")
    truth = generate_series_module_cell_profiles(
        case,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )
    module_df = aggregate_series_module(truth)
    sensed = apply_case_sensor_model(case, truth)

    report = compute_diagnostic_report(case, sensed, module_df=module_df)

    assert report.capacity_verdict == "FAIL"
    assert report.contact_R_verdict == "FAIL"
    assert report.final_verdict == "PASS_DETECTED"


def test_fault_with_no_rule_flags_is_missed_fault():
    false_positive, false_negative, final_verdict = combine_final_verdict(
        fault_label="voltage_sensor_bias",
        rule_verdict_by_name={
            "coulomb_drift_verdict": "PASS",
            "ocv_reset_verdict": "PASS",
            "voltage_residual_verdict": "PASS",
            "capacity_verdict": "PASS",
            "contact_R_verdict": "PASS",
            "module_imbalance_verdict": "PASS",
        },
    )

    assert false_positive is False
    assert false_negative is True
    assert final_verdict == "FAIL_MISSED_FAULT"


def test_wrong_rule_flag_yields_ambiguous_signature():
    false_positive, false_negative, final_verdict = combine_final_verdict(
        fault_label="voltage_sensor_bias",
        rule_verdict_by_name={
            "coulomb_drift_verdict": "PASS",
            "ocv_reset_verdict": "PASS",
            "voltage_residual_verdict": "PASS",
            "capacity_verdict": "FAIL",
            "contact_R_verdict": "PASS",
            "module_imbalance_verdict": "PASS",
        },
    )

    assert false_positive is False
    assert false_negative is False
    assert final_verdict == "FAIL_AMBIGUOUS_SIGNATURE"
