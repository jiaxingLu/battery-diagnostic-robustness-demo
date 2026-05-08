"""Tests for the manually curated v0.1 observability matrix."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bdr_demo.case_matrix import V0_1_CASES
from bdr_demo.observability import (
    DEFAULT_OBSERVABILITY_REPORT_PATH,
    V0_1_OBSERVABILITY_ROWS,
    export_observability_matrix,
    get_observability_row,
    iter_observability_rows,
    observability_to_dataframe,
    validate_observability_matrix,
)


def test_default_observability_report_path_is_repo_root_anchored():
    assert DEFAULT_OBSERVABILITY_REPORT_PATH.name == "observability_matrix.csv"
    assert DEFAULT_OBSERVABILITY_REPORT_PATH.parent.name == "reports"


def test_observability_matrix_has_one_row_per_fault_label():
    expected = {case.fault_label for case in V0_1_CASES}
    actual = {row.fault_label for row in V0_1_OBSERVABILITY_ROWS}

    assert actual == expected
    assert len(V0_1_OBSERVABILITY_ROWS) == len(expected)


def test_observability_fault_labels_are_unique():
    labels = [row.fault_label for row in V0_1_OBSERVABILITY_ROWS]
    assert len(set(labels)) == len(labels)


def test_validate_observability_matrix_passes():
    validate_observability_matrix()


def test_iter_observability_rows_matches_locked_tuple():
    assert tuple(iter_observability_rows()) == V0_1_OBSERVABILITY_ROWS


def test_get_observability_row_returns_voltage_sensor_bias():
    row = get_observability_row("voltage_sensor_bias")
    assert row.fault_label == "voltage_sensor_bias"
    assert row.ambiguity_level == "high"
    assert "10 mV" in row.notes
    assert "30 mV" in row.notes


def test_get_observability_row_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_observability_row("not_a_fault")


def test_observability_to_dataframe_exports_expected_columns():
    df = observability_to_dataframe()

    expected_columns = [
        "schema_version",
        "fault_label",
        "voltage_signature",
        "current_signature",
        "temperature_signature",
        "soc_signature",
        "soh_signature",
        "module_spread_signature",
        "ambiguity_level",
        "notes",
    ]

    assert list(df.columns) == expected_columns
    assert len(df) == len(V0_1_OBSERVABILITY_ROWS)


def test_validate_observability_matrix_rejects_duplicate_fault_label():
    duplicate_rows = list(V0_1_OBSERVABILITY_ROWS)
    duplicate_rows[-1] = duplicate_rows[0]

    with pytest.raises(ValueError, match="unique"):
        validate_observability_matrix(duplicate_rows)


def test_validate_observability_matrix_rejects_missing_fault_label():
    missing_rows = V0_1_OBSERVABILITY_ROWS[:-1]

    with pytest.raises(ValueError, match="fault_label set mismatch"):
        validate_observability_matrix(missing_rows)


def test_validate_observability_matrix_rejects_invalid_ambiguity_level():
    row = get_observability_row("none")
    bad_row = replace(row, ambiguity_level="not_valid")
    rows = [bad_row if r.fault_label == "none" else r for r in V0_1_OBSERVABILITY_ROWS]

    with pytest.raises(ValueError, match="ambiguity_level"):
        validate_observability_matrix(rows)


def test_export_observability_matrix_writes_csv(tmp_path):
    out = tmp_path / "observability_matrix.csv"
    df = export_observability_matrix(out)

    assert out.exists()
    loaded = pd.read_csv(out)
    assert loaded.shape == df.shape
    assert set(loaded["fault_label"]) == {
        row.fault_label for row in V0_1_OBSERVABILITY_ROWS
    }
