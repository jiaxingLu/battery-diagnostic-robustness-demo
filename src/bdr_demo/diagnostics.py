"""Rule-based diagnostic checks for v0.1.

This module evaluates simple demonstrator-level diagnostic rules on
sensor-layer profiles.

v0.1 scope:
    - Coulomb-counting drift proxy
    - OCV-reset consistency proxy
    - voltage residual check
    - capacity consistency check
    - contact-resistance proxy
    - module imbalance check

These checks are deliberately transparent rule-based diagnostics, not
production BMS algorithms and not model-based observers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bdr_demo.profile_generator import DEFAULT_OCV_TABLE_PATH, load_ocv_table
from bdr_demo.schema import (
    CaseSpec,
    DiagnosticReportRow,
    RuleVerdict,
    SCHEMA_VERSION,
    SensedProfileRow,
)


SOC_WARNING_THRESHOLD = 0.03
SOC_FAIL_THRESHOLD = 0.05

VOLTAGE_RESIDUAL_WARNING_MV = 30.0
VOLTAGE_RESIDUAL_FAIL_MV = 100.0

CAPACITY_FADE_FAIL_THRESHOLD = 0.05
CONTACT_R_FAIL_RATIO = 1.30

MODULE_IMBALANCE_WARNING_MV = 30.0
MODULE_IMBALANCE_FAIL_MV = 50.0


def rule_verdict_from_abs_error(
    value: float,
    *,
    warning_threshold: float,
    fail_threshold: float,
) -> RuleVerdict:
    """Convert an absolute error metric into PASS/WARNING/FAIL."""

    abs_value = abs(float(value))

    if abs_value >= fail_threshold:
        return "FAIL"

    if abs_value >= warning_threshold:
        return "WARNING"

    return "PASS"


def rule_verdict_from_upper_bound(
    value: float,
    *,
    warning_threshold: float | None = None,
    fail_threshold: float,
) -> RuleVerdict:
    """Convert an upper-bound metric into PASS/WARNING/FAIL."""

    value = float(value)

    if value > fail_threshold:
        return "FAIL"

    if warning_threshold is not None and value > warning_threshold:
        return "WARNING"

    return "PASS"


def inverse_ocv_to_soc(ocv_V: np.ndarray, ocv_table: pd.DataFrame) -> np.ndarray:
    """Map OCV values back to SOC via inverse interpolation.

    The Chen2020 OCV table exported for v0.1 is monotonic in the used range.
    Values outside the table range are clipped by numpy.interp.
    """

    table = ocv_table.sort_values("ocv_V").reset_index(drop=True)
    return np.interp(
        ocv_V,
        table["ocv_V"].to_numpy(dtype=float),
        table["soc"].to_numpy(dtype=float),
    )


def _rows_to_dataframe(rows: Sequence[SensedProfileRow]) -> pd.DataFrame:
    if not rows:
        raise ValueError("sensed_rows must not be empty")

    df = pd.DataFrame([row.__dict__ for row in rows])

    if df["case_id"].nunique() != 1:
        raise ValueError("sensed_rows must contain exactly one case_id")

    return df.sort_values(["cell_id", "t_s"]).reset_index(drop=True)


def compute_coulomb_soc_error_max(
    sensed_rows: Sequence[SensedProfileRow],
    *,
    initial_soc_assumed: float = 0.80,
    nominal_capacity_Ah: float = 3.5,
) -> float:
    """Compute maximum absolute SOC error of a simple Coulomb-counting proxy.

    Positive current denotes discharge.

    The diagnostic estimate assumes the nominal capacity and the same initial
    SOC for all cells. This deliberately exposes capacity fade and initial SOC
    inventory mismatch.
    """

    if not 0.0 <= initial_soc_assumed <= 1.0:
        raise ValueError("initial_soc_assumed must be within [0, 1]")

    if nominal_capacity_Ah <= 0:
        raise ValueError("nominal_capacity_Ah must be positive")

    df = _rows_to_dataframe(sensed_rows)

    errors: list[float] = []

    for _, group in df.groupby("cell_id", sort=True):
        group = group.sort_values("t_s").reset_index(drop=True)
        soc_est = initial_soc_assumed

        for idx, row in group.iterrows():
            errors.append(float(soc_est - row["soc_true"]))

            if idx < len(group) - 1:
                dt_s = float(group.loc[idx + 1, "t_s"] - row["t_s"])
                if dt_s < 0:
                    raise ValueError("time axis must be non-negative within each cell")
                soc_est -= float(row["current_A"]) * dt_s / (3600.0 * nominal_capacity_Ah)
                soc_est = float(np.clip(soc_est, 0.0, 1.0))

    return float(np.max(np.abs(errors)))


def compute_ocv_reset_soc_error_max(
    case: CaseSpec,
    sensed_rows: Sequence[SensedProfileRow],
    *,
    ocv_table_path: str | Path = DEFAULT_OCV_TABLE_PATH,
) -> float:
    """Compute maximum SOC error from a simple OCV-reset consistency proxy.

    v0.1 profiles currently do not include explicit rest plateaus. Therefore
    this proxy uses a compensated voltage estimate:

        OCV_est = V_measured + R0_baseline * I + V_RC

    The correction intentionally uses the baseline R0, not the true per-case
    R0. As a result, contact-resistance growth and voltage sensor bias can
    appear as OCV/SOC inconsistency.
    """

    df = _rows_to_dataframe(sensed_rows)
    ocv_table = load_ocv_table(ocv_table_path)

    ocv_est = (
        df["voltage_measured_V"].to_numpy(dtype=float)
        + case.r0_baseline_Ohm * df["current_A"].to_numpy(dtype=float)
        + df["v_rc_V"].to_numpy(dtype=float)
    )
    soc_est = inverse_ocv_to_soc(ocv_est, ocv_table)

    soc_true = df["soc_true"].to_numpy(dtype=float)
    return float(np.max(np.abs(soc_est - soc_true)))


def compute_voltage_residual_max_mV(
    sensed_rows: Sequence[SensedProfileRow],
) -> float:
    """Compute maximum absolute sensor-layer voltage residual in mV."""

    df = _rows_to_dataframe(sensed_rows)
    residual_mV = 1000.0 * (
        df["voltage_measured_V"].to_numpy(dtype=float)
        - df["voltage_true_V"].to_numpy(dtype=float)
    )
    return float(np.max(np.abs(residual_mV)))


def compute_capacity_consistency_error(
    case: CaseSpec,
    sensed_rows: Sequence[SensedProfileRow],
) -> float:
    """Compute capacity consistency error.

    Defined as:

        1 - mean(C_effective) / C_nominal

    Positive values indicate effective capacity reduction.
    """

    df = _rows_to_dataframe(sensed_rows)
    capacity_mean = float(df.groupby("cell_id")["capacity_Ah"].first().mean())
    return float(1.0 - capacity_mean / case.nominal_capacity_Ah)


def compute_contact_R_inferred_ratio(
    case: CaseSpec,
    sensed_rows: Sequence[SensedProfileRow],
) -> float:
    """Compute contact-resistance proxy ratio.

    Defined as:

        mean(R0_effective) / R0_baseline
    """

    df = _rows_to_dataframe(sensed_rows)
    r0_mean = float(df.groupby("cell_id")["r0_Ohm"].first().mean())
    return float(r0_mean / case.r0_baseline_Ohm)


def compute_module_delta_voltage_max_mV(
    module_df: pd.DataFrame | None,
) -> float:
    """Extract maximum module cell-voltage spread from module_df.

    If no module aggregation is provided, returns 0.0.
    """

    if module_df is None:
        return 0.0

    if module_df.empty:
        return 0.0

    if "delta_cell_voltage_mV" not in module_df.columns:
        raise ValueError("module_df must contain delta_cell_voltage_mV")

    return float(module_df["delta_cell_voltage_mV"].max())


def combine_final_verdict(
    *,
    fault_label: str,
    rule_verdicts: Sequence[RuleVerdict],
    metrics_nonzero_but_below_threshold: bool,
) -> tuple[bool, bool, str]:
    """Combine rule verdicts into final diagnostic robustness verdict.

    Returns:
        ``(false_positive, false_negative, final_verdict)``.
    """

    has_warning = any(verdict == "WARNING" for verdict in rule_verdicts)
    has_fail = any(verdict == "FAIL" for verdict in rule_verdicts)
    has_any_flag = has_warning or has_fail

    if fault_label == "none":
        if has_any_flag:
            return True, False, "FAIL_FALSE_ALARM"
        return False, False, "PASS_HEALTHY"

    if has_fail:
        return False, False, "PASS_DETECTED"

    if has_warning:
        return False, False, "WARNING_BOUNDARY"

    if metrics_nonzero_but_below_threshold:
        return False, True, "FAIL_AMBIGUOUS_SIGNATURE"

    return False, True, "FAIL_MISSED_FAULT"


def compute_diagnostic_report(
    case: CaseSpec,
    sensed_rows: Sequence[SensedProfileRow],
    *,
    module_df: pd.DataFrame | None = None,
    initial_soc_assumed: float = 0.80,
    ocv_table_path: str | Path = DEFAULT_OCV_TABLE_PATH,
) -> DiagnosticReportRow:
    """Compute one diagnostic report row for a case."""

    df = _rows_to_dataframe(sensed_rows)

    if set(df["case_id"]) != {case.case_id}:
        raise ValueError(
            f"sensed_rows case_id does not match case: {set(df['case_id'])} != {case.case_id}"
        )

    soc_error_coulomb_max = compute_coulomb_soc_error_max(
        sensed_rows,
        initial_soc_assumed=initial_soc_assumed,
        nominal_capacity_Ah=case.nominal_capacity_Ah,
    )
    soc_error_ocv_reset_max = compute_ocv_reset_soc_error_max(
        case,
        sensed_rows,
        ocv_table_path=ocv_table_path,
    )
    voltage_residual_max_mV = compute_voltage_residual_max_mV(sensed_rows)
    capacity_consistency_error = compute_capacity_consistency_error(case, sensed_rows)
    contact_R_inferred_ratio = compute_contact_R_inferred_ratio(case, sensed_rows)
    module_delta_voltage_max_mV = compute_module_delta_voltage_max_mV(module_df)

    coulomb_drift_verdict = rule_verdict_from_abs_error(
        soc_error_coulomb_max,
        warning_threshold=SOC_WARNING_THRESHOLD,
        fail_threshold=SOC_FAIL_THRESHOLD,
    )
    ocv_reset_verdict = rule_verdict_from_abs_error(
        soc_error_ocv_reset_max,
        warning_threshold=SOC_WARNING_THRESHOLD,
        fail_threshold=SOC_FAIL_THRESHOLD,
    )
    voltage_residual_verdict = rule_verdict_from_upper_bound(
        voltage_residual_max_mV,
        warning_threshold=VOLTAGE_RESIDUAL_WARNING_MV,
        fail_threshold=VOLTAGE_RESIDUAL_FAIL_MV,
    )
    capacity_verdict = rule_verdict_from_upper_bound(
        capacity_consistency_error,
        warning_threshold=None,
        fail_threshold=CAPACITY_FADE_FAIL_THRESHOLD,
    )
    contact_R_verdict = rule_verdict_from_upper_bound(
        contact_R_inferred_ratio,
        warning_threshold=None,
        fail_threshold=CONTACT_R_FAIL_RATIO,
    )
    module_imbalance_verdict = rule_verdict_from_upper_bound(
        module_delta_voltage_max_mV,
        warning_threshold=MODULE_IMBALANCE_WARNING_MV,
        fail_threshold=MODULE_IMBALANCE_FAIL_MV,
    )

    rule_verdicts: tuple[RuleVerdict, ...] = (
        coulomb_drift_verdict,
        ocv_reset_verdict,
        voltage_residual_verdict,
        capacity_verdict,
        contact_R_verdict,
        module_imbalance_verdict,
    )

    # Used to distinguish a complete miss from an intentionally ambiguous
    # below-threshold signature, e.g. +10 mV voltage sensor bias under a
    # 30 mV warning threshold.
    metrics_nonzero_but_below_threshold = bool(
        voltage_residual_max_mV > 0.0
        or abs(case.initial_soc_mismatch) > 0.0
        or abs(capacity_consistency_error) > 0.0
        or abs(contact_R_inferred_ratio - 1.0) > 0.0
        or module_delta_voltage_max_mV > 0.0
    )

    false_positive, false_negative, final_verdict = combine_final_verdict(
        fault_label=case.fault_label,
        rule_verdicts=rule_verdicts,
        metrics_nonzero_but_below_threshold=metrics_nonzero_but_below_threshold,
    )

    return DiagnosticReportRow(
        schema_version=SCHEMA_VERSION,
        case_id=case.case_id,
        fault_label=case.fault_label,
        soc_error_coulomb_max=soc_error_coulomb_max,
        soc_error_ocv_reset_max=soc_error_ocv_reset_max,
        voltage_residual_max_mV=voltage_residual_max_mV,
        capacity_consistency_error=capacity_consistency_error,
        contact_R_inferred_ratio=contact_R_inferred_ratio,
        module_delta_voltage_max_mV=module_delta_voltage_max_mV,
        coulomb_drift_verdict=coulomb_drift_verdict,
        ocv_reset_verdict=ocv_reset_verdict,
        voltage_residual_verdict=voltage_residual_verdict,
        capacity_verdict=capacity_verdict,
        contact_R_verdict=contact_R_verdict,
        module_imbalance_verdict=module_imbalance_verdict,
        false_positive=false_positive,
        false_negative=false_negative,
        final_verdict=final_verdict,  # type: ignore[arg-type]
    )


__all__ = [
    "SOC_WARNING_THRESHOLD",
    "SOC_FAIL_THRESHOLD",
    "VOLTAGE_RESIDUAL_WARNING_MV",
    "VOLTAGE_RESIDUAL_FAIL_MV",
    "CAPACITY_FADE_FAIL_THRESHOLD",
    "CONTACT_R_FAIL_RATIO",
    "MODULE_IMBALANCE_WARNING_MV",
    "MODULE_IMBALANCE_FAIL_MV",
    "rule_verdict_from_abs_error",
    "rule_verdict_from_upper_bound",
    "inverse_ocv_to_soc",
    "compute_coulomb_soc_error_max",
    "compute_ocv_reset_soc_error_max",
    "compute_voltage_residual_max_mV",
    "compute_capacity_consistency_error",
    "compute_contact_R_inferred_ratio",
    "compute_module_delta_voltage_max_mV",
    "combine_final_verdict",
    "compute_diagnostic_report",
]
