"""Tests for the v0.1 synthetic 1RC profile generator."""

from __future__ import annotations

import numpy as np

from bdr_demo.case_matrix import get_case
from bdr_demo.profile_generator import (
    DEFAULT_OCV_TABLE_PATH,
    generate_1rc_profile,
    interpolate_ocv,
    load_ocv_table,
    make_time_grid,
    profiles_to_dataframe,
)
from bdr_demo.schema import ProfileRow, validate_profile_row


def test_load_ocv_table_passes_static_csv():
    df = load_ocv_table(DEFAULT_OCV_TABLE_PATH)
    assert set(["soc", "ocv_V"]).issubset(df.columns)
    assert len(df) == 201
    assert df["soc"].is_monotonic_increasing
    assert df["ocv_V"].between(2.0, 5.0).all()


def test_interpolate_ocv_returns_plausible_value():
    table = load_ocv_table(DEFAULT_OCV_TABLE_PATH)
    ocv_mid = float(interpolate_ocv(0.5, table))
    assert 3.0 < ocv_mid < 4.2


def test_make_time_grid_is_inclusive():
    grid = make_time_grid(t_end_s=10.0, dt_s=1.0)
    assert len(grid) == 11
    assert grid[0] == 0.0
    assert grid[-1] == 10.0


def test_generate_baseline_profile_rows_validate():
    case = get_case("baseline")
    rows = generate_1rc_profile(
        case,
        cell_id=1,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=60.0,
        dt_s=1.0,
    )

    assert len(rows) == 61
    assert all(isinstance(row, ProfileRow) for row in rows)

    for row in rows:
        validate_profile_row(row)


def test_profile_time_is_strictly_increasing():
    case = get_case("baseline")
    rows = generate_1rc_profile(case, t_end_s=60.0, dt_s=1.0)
    t = np.array([row.t_s for row in rows])
    assert np.all(np.diff(t) > 0)


def test_soc_decreases_under_positive_discharge_current():
    case = get_case("baseline")
    rows = generate_1rc_profile(
        case,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )

    assert rows[-1].soc_true < rows[0].soc_true


def test_zero_current_profile_keeps_soc_constant():
    case = get_case("baseline")
    rows = generate_1rc_profile(
        case,
        initial_soc_assumed=0.8,
        discharge_current_A=0.0,
        t_end_s=60.0,
        dt_s=1.0,
    )

    assert rows[-1].soc_true == rows[0].soc_true


def test_capacity_fade_changes_soc_slope():
    baseline = get_case("baseline")
    faded = get_case("capacity_fade_90")

    rows_base = generate_1rc_profile(
        baseline,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )
    rows_faded = generate_1rc_profile(
        faded,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=600.0,
        dt_s=1.0,
    )

    assert rows_faded[-1].soc_true < rows_base[-1].soc_true


def test_contact_resistance_growth_lowers_under_load_voltage():
    baseline = get_case("baseline")
    high_r = get_case("contact_resistance_150")

    rows_base = generate_1rc_profile(
        baseline,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=10.0,
        dt_s=1.0,
    )
    rows_high_r = generate_1rc_profile(
        high_r,
        initial_soc_assumed=0.8,
        discharge_current_A=1.0,
        t_end_s=10.0,
        dt_s=1.0,
    )

    assert rows_high_r[0].voltage_true_V < rows_base[0].voltage_true_V


def test_initial_soc_mismatch_changes_true_initial_soc():
    baseline = get_case("baseline")
    mismatch = get_case("initial_soc_mismatch_p05")

    rows_base = generate_1rc_profile(
        baseline,
        initial_soc_assumed=0.8,
        discharge_current_A=0.0,
        t_end_s=1.0,
        dt_s=1.0,
    )
    rows_mismatch = generate_1rc_profile(
        mismatch,
        initial_soc_assumed=0.8,
        discharge_current_A=0.0,
        t_end_s=1.0,
        dt_s=1.0,
    )

    assert rows_mismatch[0].soc_true == rows_base[0].soc_true + 0.05


def test_profiles_to_dataframe_exports_profile_columns():
    case = get_case("baseline")
    rows = generate_1rc_profile(case, t_end_s=5.0, dt_s=1.0)
    df = profiles_to_dataframe(rows)

    expected = [
        "schema_version",
        "case_id",
        "level",
        "cell_id",
        "t_s",
        "current_A",
        "voltage_true_V",
        "ocv_V",
        "soc_true",
        "v_rc_V",
    ]
    assert list(df.columns)[: len(expected)] == expected
    assert len(df) == 6
