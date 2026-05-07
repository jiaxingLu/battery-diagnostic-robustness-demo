"""Tests for 24s1p pure-series module aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from bdr_demo.case_matrix import N_SERIES_CELLS, get_case
from bdr_demo.module_aggregator import (
    aggregate_series_module,
    generate_series_module_cell_profiles,
    make_deterministic_scatter,
    make_module_inconsistency_report,
)
from bdr_demo.schema import ModuleInconsistencyReportRow


def test_deterministic_scatter_spans_requested_range():
    scatter = make_deterministic_scatter(5, 0.05)
    assert np.isclose(scatter[0], -0.05)
    assert np.isclose(scatter[-1], 0.05)
    assert np.isclose(scatter.mean(), 0.0)


def test_deterministic_scatter_zero_returns_zeros():
    scatter = make_deterministic_scatter(4, 0.0)
    assert np.allclose(scatter, np.zeros(4))


def test_generate_series_module_baseline_has_24_cells():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)

    cell_ids = {row.cell_id for row in rows}
    assert len(cell_ids) == N_SERIES_CELLS
    assert cell_ids == set(range(1, N_SERIES_CELLS + 1))


def test_generate_series_module_rows_count_matches_cells_times_steps():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)

    assert len(rows) == N_SERIES_CELLS * 11


def test_uniform_baseline_aggregation_equals_24_times_cell_voltage():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)

    first_cell_v = rows[0].voltage_true_V
    first_pack_v = float(module_df.loc[module_df["t_s"] == 0.0, "pack_voltage_true_V"].iloc[0])

    assert np.isclose(first_pack_v, N_SERIES_CELLS * first_cell_v)
    assert float(module_df["delta_cell_voltage_mV"].max()) == 0.0


def test_module_soc_scatter_creates_cell_voltage_spread():
    case = get_case("module_soc_scatter")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)

    assert float(module_df["delta_cell_voltage_mV"].max()) > 0.0


def test_module_capacity_scatter_creates_capacity_spread():
    case = get_case("module_capacity_scatter")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)

    capacities = {round(row.capacity_Ah, 6) for row in rows}
    assert len(capacities) > 1


def test_combined_fault_creates_soc_and_capacity_spread():
    case = get_case("combined_fault")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)

    initial_rows = [row for row in rows if row.t_s == 0.0]
    socs = {round(row.soc_true, 6) for row in initial_rows}
    capacities = {round(row.capacity_Ah, 6) for row in rows}

    assert len(socs) > 1
    assert len(capacities) > 1


def test_aggregate_series_module_returns_expected_columns():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=5.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)

    expected = [
        "schema_version",
        "case_id",
        "t_s",
        "current_A",
        "pack_voltage_true_V",
        "min_cell_voltage_V",
        "max_cell_voltage_V",
        "delta_cell_voltage_mV",
        "n_cells",
    ]
    assert list(module_df.columns) == expected
    assert len(module_df) == 6
    assert set(module_df["n_cells"]) == {N_SERIES_CELLS}


def test_module_inconsistency_report_instantiates_from_baseline():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=10.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)
    report = make_module_inconsistency_report(rows, module_df)

    assert isinstance(report, ModuleInconsistencyReportRow)
    assert report.case_id == "baseline"
    assert report.weakest_cell_id in range(1, N_SERIES_CELLS + 1)
    assert report.module_risk_flag == "PASS"


def test_module_inconsistency_report_detects_soc_scatter_risk():
    case = get_case("module_soc_scatter")
    rows = generate_series_module_cell_profiles(case, t_end_s=60.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)
    report = make_module_inconsistency_report(rows, module_df)

    assert report.delta_cell_voltage_max_mV > 0.0
    assert report.module_risk_flag in {"WARNING", "FAIL"}


def test_aggregate_series_module_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        aggregate_series_module([])


def test_module_report_rejects_unknown_weakest_criterion():
    case = get_case("baseline")
    rows = generate_series_module_cell_profiles(case, t_end_s=1.0, dt_s=1.0)
    module_df = aggregate_series_module(rows)

    with pytest.raises(ValueError, match="Unknown weakest_cell_criterion"):
        make_module_inconsistency_report(
            rows,
            module_df,
            weakest_cell_criterion="not_a_real_criterion",
        )
