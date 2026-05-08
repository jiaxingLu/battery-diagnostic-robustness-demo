"""Tests for the v0.1 end-to-end report pipeline."""

from __future__ import annotations

import pandas as pd

from bdr_demo.case_matrix import V0_1_CASES
from bdr_demo.pipeline import run_all_cases, run_case, run_pipeline
from bdr_demo.schema import DiagnosticReportRow, ModuleInconsistencyReportRow


def test_run_case_returns_report_rows():
    case = V0_1_CASES[0]
    diagnostic_row, module_row, module_df = run_case(
        case,
        t_end_s=20.0,
        dt_s=1.0,
    )

    assert isinstance(diagnostic_row, DiagnosticReportRow)
    assert isinstance(module_row, ModuleInconsistencyReportRow)
    assert not module_df.empty
    assert diagnostic_row.case_id == case.case_id
    assert module_row.case_id == case.case_id


def test_run_all_cases_returns_eight_report_rows():
    diagnostic_df, module_df = run_all_cases(t_end_s=20.0, dt_s=1.0)

    assert len(diagnostic_df) == 8
    assert len(module_df) == 8
    assert set(diagnostic_df["case_id"]) == {case.case_id for case in V0_1_CASES}
    assert set(module_df["case_id"]) == {case.case_id for case in V0_1_CASES}


def test_pipeline_expected_key_verdicts():
    diagnostic_df, _ = run_all_cases(t_end_s=600.0, dt_s=1.0)
    by_case = diagnostic_df.set_index("case_id")

    assert by_case.loc["baseline", "final_verdict"] == "PASS_HEALTHY"
    assert by_case.loc["capacity_fade_90", "final_verdict"] == "PASS_DETECTED"
    assert by_case.loc["contact_resistance_150", "final_verdict"] == "PASS_DETECTED"
    assert by_case.loc["initial_soc_mismatch_p05", "final_verdict"] == "PASS_DETECTED"
    assert (
        by_case.loc["voltage_bias_p10mV", "final_verdict"]
        == "FAIL_MISSED_FAULT"
    )
    assert (
        by_case.loc["module_capacity_scatter", "final_verdict"]
        == "FAIL_MISSED_FAULT"
    )
    assert by_case.loc["combined_fault", "final_verdict"] == "PASS_DETECTED"


def test_run_pipeline_writes_reports_and_plots(tmp_path):
    diagnostic_df, module_df = run_pipeline(
        report_dir=tmp_path,
        make_plots=True,
        t_end_s=20.0,
        dt_s=1.0,
    )

    diagnostic_path = tmp_path / "diagnostic_robustness_report.csv"
    module_path = tmp_path / "module_inconsistency_report.csv"
    observability_path = tmp_path / "observability_matrix.csv"

    assert diagnostic_path.exists()
    assert module_path.exists()
    assert observability_path.exists()

    loaded_diag = pd.read_csv(diagnostic_path)
    loaded_module = pd.read_csv(module_path)
    loaded_obs = pd.read_csv(observability_path)

    assert loaded_diag.shape == diagnostic_df.shape
    assert loaded_module.shape == module_df.shape
    assert len(loaded_obs) == 8

    plot_dir = tmp_path / "plots"
    assert (plot_dir / "diagnostic_verdict_counts.png").exists()
    assert (plot_dir / "rule_verdict_matrix.png").exists()
    assert (plot_dir / "voltage_residual_max_mV.png").exists()
