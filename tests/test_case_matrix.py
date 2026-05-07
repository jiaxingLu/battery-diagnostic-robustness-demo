"""Tests for the v0.1 diagnostic perturbation case matrix."""

from __future__ import annotations

import pytest

from bdr_demo.case_matrix import (
    N_SERIES_CELLS,
    V0_1_CASES,
    get_case,
    iter_cases,
    validate_case_matrix,
)
from bdr_demo.schema import CaseSpec, validate_case_spec


def test_case_matrix_contains_eight_cases():
    assert len(V0_1_CASES) == 8


def test_case_matrix_ids_are_unique():
    case_ids = [case.case_id for case in V0_1_CASES]
    assert len(set(case_ids)) == len(case_ids)


def test_case_matrix_required_ids_are_present():
    case_ids = {case.case_id for case in V0_1_CASES}
    assert case_ids == {
        "baseline",
        "capacity_fade_90",
        "contact_resistance_150",
        "initial_soc_mismatch_p05",
        "voltage_bias_p10mV",
        "module_soc_scatter",
        "module_capacity_scatter",
        "combined_fault",
    }


def test_each_case_spec_validates():
    for case in V0_1_CASES:
        validate_case_spec(case)


def test_validate_case_matrix_passes():
    validate_case_matrix()


def test_get_case_returns_requested_case():
    case = get_case("contact_resistance_150")
    assert case.case_id == "contact_resistance_150"
    assert case.fault_label == "contact_resistance_growth"
    assert case.contact_resistance_factor == 1.5


def test_get_case_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_case("does_not_exist")


def test_iter_cases_matches_locked_tuple():
    assert tuple(iter_cases()) == V0_1_CASES


def test_series_cell_count_locked_to_24():
    assert N_SERIES_CELLS == 24


def test_baseline_case_is_healthy_reference():
    case = get_case("baseline")
    assert case.fault_label == "none"
    assert case.capacity_factor == 1.0
    assert case.contact_resistance_factor == 1.0
    assert case.initial_soc_mismatch == 0.0
    assert case.voltage_bias_mV == 0.0
    assert case.cell_scatter_type == "none"
    assert case.cell_scatter_magnitude == 0.0


def test_initial_soc_mismatch_case_uses_locked_nomenclature():
    case = get_case("initial_soc_mismatch_p05")
    assert case.fault_label == "initial_soc_inventory_offset"
    assert case.initial_soc_mismatch == 0.05


def test_voltage_bias_case_is_10_mV():
    case = get_case("voltage_bias_p10mV")
    assert case.voltage_bias_mV == 10.0
    assert case.fault_label == "voltage_sensor_bias"


def test_module_scatter_cases_are_module_level():
    soc_case = get_case("module_soc_scatter")
    capacity_case = get_case("module_capacity_scatter")
    combined_case = get_case("combined_fault")

    assert soc_case.level == "module"
    assert soc_case.cell_scatter_type == "soc"
    assert soc_case.cell_scatter_magnitude == 0.05

    assert capacity_case.level == "module"
    assert capacity_case.cell_scatter_type == "capacity"
    assert capacity_case.cell_scatter_magnitude == 0.05

    assert combined_case.level == "module"
    assert combined_case.cell_scatter_type == "mixed"
    assert combined_case.cell_scatter_magnitude == 0.05


def test_uniform_cell_cases_have_no_explicit_scatter():
    for case_id in [
        "baseline",
        "capacity_fade_90",
        "contact_resistance_150",
        "initial_soc_mismatch_p05",
        "voltage_bias_p10mV",
    ]:
        case = get_case(case_id)
        assert case.level == "cell"
        assert case.cell_scatter_type == "none"
        assert case.cell_scatter_magnitude == 0.0


def test_validate_case_matrix_rejects_missing_case():
    with pytest.raises(ValueError, match="8 cases"):
        validate_case_matrix(V0_1_CASES[:-1])


def test_validate_case_matrix_rejects_duplicate_ids():
    duplicate = CaseSpec(
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
    cases = tuple(list(V0_1_CASES[:-1]) + [duplicate])
    with pytest.raises(ValueError, match="unique"):
        validate_case_matrix(cases)
