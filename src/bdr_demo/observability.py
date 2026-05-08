"""Manually curated observability matrix for v0.1.

This module documents which signal channels are expected to carry evidence
for each controlled fault label in the v0.1 case matrix.

Important:
    The v0.1 observability matrix is a manually curated design artifact.
    It is not generated from automated feature extraction.

Purpose:
    - document expected fault signatures;
    - expose ambiguity between fault classes;
    - support signal-to-state diagnostic interpretation.

Side effects:
    Importing this module triggers ``validate_observability_matrix()``
    against the locked ``V0_1_OBSERVABILITY_ROWS`` tuple. Any drift between
    the manual matrix and the case matrix raises ``ValueError`` at import
    time. This is intentional fail-fast behaviour per contract §8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from bdr_demo.case_matrix import V0_1_CASES
from bdr_demo.schema import (
    ObservabilityRow,
    SCHEMA_VERSION,
    rows_to_dataframe,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OBSERVABILITY_REPORT_PATH = _REPO_ROOT / "reports" / "observability_matrix.csv"

_VALID_AMBIGUITY_LEVELS = ("low", "medium", "high")


V0_1_OBSERVABILITY_ROWS: tuple[ObservabilityRow, ...] = (
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="none",
        voltage_signature="reference voltage trajectory",
        current_signature="reference prescribed current",
        temperature_signature="not modelled in v0.1",
        soc_signature="reference SOC trajectory",
        soh_signature="reference capacity and resistance",
        module_spread_signature="no intentional cell-to-cell spread",
        ambiguity_level="low",
        notes=(
            "Healthy reference case. Any diagnostic flag in this case is treated "
            "as a false alarm."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="capacity_fade",
        voltage_signature=(
            "terminal voltage may decline faster during discharge because SOC "
            "depletes faster under the same current"
        ),
        current_signature="no direct current-channel signature under prescribed current",
        temperature_signature="not modelled in v0.1",
        soc_signature=(
            "Coulomb-counting estimate based on nominal capacity drifts relative "
            "to true SOC"
        ),
        soh_signature="effective capacity below nominal capacity",
        module_spread_signature=(
            "none for uniform capacity fade; possible spread only in module-level "
            "capacity-inconsistency cases"
        ),
        ambiguity_level="medium",
        notes=(
            "Can overlap with initial SOC mismatch in SOC-error metrics. Capacity "
            "consistency is the primary disambiguating criterion."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="contact_resistance_growth",
        voltage_signature=(
            "under-load terminal voltage is shifted lower at identical current "
            "and SOC due to increased ohmic drop"
        ),
        current_signature="no direct current-channel signature under prescribed current",
        temperature_signature="not modelled in v0.1",
        soc_signature="SOC trajectory remains close to baseline if capacity is unchanged",
        soh_signature="contact-resistance proxy ratio exceeds baseline",
        module_spread_signature=(
            "none for uniform resistance growth; spread would require cell-specific R0 scatter"
        ),
        ambiguity_level="medium",
        notes=(
            "Can resemble low SOC or voltage depression in the voltage channel. "
            "R0/contact-resistance proxy is required for disambiguation."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="initial_soc_inventory_offset",
        voltage_signature=(
            "initial voltage is shifted because true initial SOC differs from the "
            "diagnostic baseline"
        ),
        current_signature="no direct current-channel signature",
        temperature_signature="not modelled in v0.1",
        soc_signature=(
            "Coulomb-counting estimate starts from the assumed SOC and therefore "
            "remains offset from true SOC"
        ),
        soh_signature="capacity and resistance remain nominal",
        module_spread_signature=(
            "may create cell-voltage spread if mismatch is applied non-uniformly"
        ),
        ambiguity_level="medium",
        notes=(
            "Can be confused with capacity fade if only SOC error is inspected. "
            "Capacity consistency helps separate inventory offset from SOH loss."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="voltage_sensor_bias",
        voltage_signature="measured voltage is shifted relative to true terminal voltage",
        current_signature="no current-channel signature",
        temperature_signature="not modelled in v0.1",
        soc_signature="OCV-derived SOC may shift if measured voltage is used for reset",
        soh_signature="capacity and resistance remain physically unchanged",
        module_spread_signature=(
            "none if the same voltage bias is applied uniformly; channel-specific "
            "bias would require a later version"
        ),
        ambiguity_level="high",
        notes=(
            "The v0.1 +10 mV bias is intentionally below the 30 mV warning threshold. "
            "This is an ambiguity case: a true deviation exists but the rule-based "
            "diagnostic may not flag it."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="cell_imbalance",
        voltage_signature=(
            "individual cell voltages separate even when pack current is identical"
        ),
        current_signature="no current redistribution in 24s1p v0.1 topology",
        temperature_signature="not modelled in v0.1",
        soc_signature="cell-level SOC values differ across the series string",
        soh_signature="capacity and resistance may remain nominal",
        module_spread_signature="max-min cell voltage spread increases",
        ambiguity_level="medium",
        notes=(
            "Main evidence is module-level voltage spread. It can overlap with "
            "capacity inconsistency if only voltage spread is inspected."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="capacity_inconsistency",
        voltage_signature=(
            "cell voltages diverge progressively because cells deplete at different rates"
        ),
        current_signature="same series current through all cells",
        temperature_signature="not modelled in v0.1",
        soc_signature="SOC trajectories diverge due to different effective capacities",
        soh_signature="cell-level effective capacities differ",
        module_spread_signature="cell-voltage spread increases over time",
        ambiguity_level="medium",
        notes=(
            "Can resemble SOC imbalance in voltage-spread metrics. Capacity metadata "
            "or capacity-consistency evidence is needed for disambiguation."
        ),
    ),
    ObservabilityRow(
        schema_version=SCHEMA_VERSION,
        fault_label="combined",
        voltage_signature=(
            "mixed voltage signatures from capacity fade, contact-resistance growth, "
            "SOC mismatch, sensor bias, and cell scatter"
        ),
        current_signature="same prescribed current; no redistribution in v0.1",
        temperature_signature="not modelled in v0.1",
        soc_signature="SOC drift and cell-to-cell SOC spread may coexist",
        soh_signature="capacity loss and resistance growth may coexist",
        module_spread_signature="cell-voltage spread likely increases",
        ambiguity_level="high",
        notes=(
            "Combined case is expected to be multi-causal. v0.1 diagnostics should "
            "identify that multiple evidence channels are active rather than claim "
            "a single root cause."
        ),
    ),
)


def iter_observability_rows() -> Iterable[ObservabilityRow]:
    """Return an iterator over the locked v0.1 observability rows."""

    return iter(V0_1_OBSERVABILITY_ROWS)


def get_observability_row(fault_label: str) -> ObservabilityRow:
    """Return one observability row by fault label."""

    for row in V0_1_OBSERVABILITY_ROWS:
        if row.fault_label == fault_label:
            return row

    raise KeyError(f"Unknown fault_label: {fault_label}")


def validate_observability_matrix(
    rows: Iterable[ObservabilityRow] = V0_1_OBSERVABILITY_ROWS,
) -> None:
    """Validate the manually curated v0.1 observability matrix."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("observability matrix must not be empty")

    fault_labels = [row.fault_label for row in row_list]
    if len(set(fault_labels)) != len(fault_labels):
        raise ValueError(f"fault_label values must be unique, got: {fault_labels}")

    expected_fault_labels = {case.fault_label for case in V0_1_CASES}
    actual_fault_labels = set(fault_labels)

    missing = expected_fault_labels - actual_fault_labels
    extra = actual_fault_labels - expected_fault_labels
    if missing or extra:
        raise ValueError(
            f"fault_label set mismatch: missing={missing}, extra={extra}"
        )

    for row in row_list:
        if row.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"invalid schema_version for {row.fault_label}: "
                f"{row.schema_version}"
            )

        if row.ambiguity_level not in _VALID_AMBIGUITY_LEVELS:
            raise ValueError(
                f"invalid ambiguity_level for {row.fault_label}: "
                f"{row.ambiguity_level}"
            )

        for field_name in (
            "voltage_signature",
            "current_signature",
            "temperature_signature",
            "soc_signature",
            "soh_signature",
            "module_spread_signature",
            "notes",
        ):
            value = getattr(row, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string for {row.fault_label}"
                )


def observability_to_dataframe(
    rows: Iterable[ObservabilityRow] = V0_1_OBSERVABILITY_ROWS,
) -> pd.DataFrame:
    """Convert observability rows into a DataFrame."""

    row_list = list(rows)
    validate_observability_matrix(row_list)
    return rows_to_dataframe(row_list)


def export_observability_matrix(
    path: str | Path = DEFAULT_OBSERVABILITY_REPORT_PATH,
) -> pd.DataFrame:
    """Export the v0.1 observability matrix to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = observability_to_dataframe()
    df.to_csv(out_path, index=False)

    print(f"[OK] saved observability matrix: {out_path}")
    print(f"[rows] {len(df)}")
    return df


# Fail early at import time if the manual matrix drifts out of contract.
validate_observability_matrix()


__all__ = [
    "DEFAULT_OBSERVABILITY_REPORT_PATH",
    "V0_1_OBSERVABILITY_ROWS",
    "iter_observability_rows",
    "get_observability_row",
    "validate_observability_matrix",
    "observability_to_dataframe",
    "export_observability_matrix",
]
