"""Schema dataclasses for battery-diagnostic-robustness-demo v0.1.

This module defines the data contracts used by the v0.1 signal-to-state
diagnostic audit pipeline. It intentionally contains no simulation logic.

All dataclasses are frozen and validated against the v0.1 methodology
contract (docs/methodology_decisions.md). Field names, types, and the
closed enumerations below are part of the contract: changing them requires
a contract revision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Literal

import pandas as pd


SCHEMA_VERSION = "bdr_demo_v0_1"


# ---------------------------------------------------------------------------
# Closed enumerations of the v0.1 contract
# ---------------------------------------------------------------------------

CaseLevel = Literal["cell", "module"]

CellScatterType = Literal["none", "soc", "capacity", "mixed"]

FaultLabel = Literal[
    "none",
    "capacity_fade",
    "contact_resistance_growth",
    "initial_soc_inventory_offset",
    "voltage_sensor_bias",
    "cell_imbalance",
    "capacity_inconsistency",
    "combined",
]

RuleVerdict = Literal["PASS", "WARNING", "FAIL"]

FinalVerdict = Literal[
    "PASS_HEALTHY",
    "PASS_DETECTED",
    "WARNING_BOUNDARY",
    "FAIL_MISSED_FAULT",
    "FAIL_FALSE_ALARM",
    "FAIL_AMBIGUOUS_SIGNATURE",
]

WeakestCriterion = Literal[
    "lowest_capacity",
    "lowest_min_voltage",
    "highest_R0",
]


# Tuples used for runtime validation. Python's Literal types are static
# annotations only and are not enforced at runtime; these tuples mirror
# the corresponding Literal definitions so validators can produce clear
# ValueError messages.
_VALID_LEVELS = ("cell", "module")
_VALID_CELL_SCATTER_TYPES = ("none", "soc", "capacity", "mixed")
_VALID_FAULT_LABELS = (
    "none",
    "capacity_fade",
    "contact_resistance_growth",
    "initial_soc_inventory_offset",
    "voltage_sensor_bias",
    "cell_imbalance",
    "capacity_inconsistency",
    "combined",
)
_VALID_RULE_VERDICTS = ("PASS", "WARNING", "FAIL")
_VALID_FINAL_VERDICTS = (
    "PASS_HEALTHY",
    "PASS_DETECTED",
    "WARNING_BOUNDARY",
    "FAIL_MISSED_FAULT",
    "FAIL_FALSE_ALARM",
    "FAIL_AMBIGUOUS_SIGNATURE",
)
_VALID_WEAKEST_CRITERIA = (
    "lowest_capacity",
    "lowest_min_voltage",
    "highest_R0",
)


# ---------------------------------------------------------------------------
# Case-specification dataclass: input layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    """Controlled perturbation case definition.

    Per methodology contract §6, ``initial_soc_mismatch`` represents a
    physical-layer disagreement between the cell's true initial SOC and
    the BMS-assumed initial SOC. It is not an estimator algorithm bug.

    A positive value means the true SOC exceeds the BMS-assumed SOC by
    that fraction. Example: ``+0.05`` means true SOC is 5 percentage
    points higher than assumed.

    ``cell_scatter_magnitude`` is a dimensionless fraction. Example:
    ``0.05`` encodes ±5 %. ``cell_scatter_type`` selects which physical
    parameter the scatter is applied to.

    Cross-field validation rules:

    - ``cell_scatter_type == "none"`` requires ``magnitude == 0``.
    - ``cell_scatter_type != "none"`` requires ``magnitude > 0``.
    - cell-level cases must have ``cell_scatter_type == "none"``.
    """

    case_id: str
    level: CaseLevel
    capacity_factor: float
    contact_resistance_factor: float
    initial_soc_mismatch: float
    voltage_bias_mV: float
    cell_scatter_type: CellScatterType
    cell_scatter_magnitude: float
    fault_label: FaultLabel
    nominal_capacity_Ah: float = 3.5
    nominal_voltage_V: float = 3.6
    r0_baseline_Ohm: float = 0.05
    r1_baseline_Ohm: float = 0.02
    c1_baseline_F: float = 500.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Profile rows: truth layer and sensor layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileRow:
    """Ground-truth cell-level profile row.

    Sign convention: positive ``current_A`` denotes discharge, per
    methodology contract §2.

    ``capacity_Ah`` is the effective per-cell capacity after applying
    ``capacity_factor`` to the baseline nominal capacity.

    ``r0_Ohm``, ``r1_Ohm``, and ``c1_F`` are effective values after
    applying perturbations. Baseline values are stored on the associated
    :class:`CaseSpec`.
    """

    schema_version: str
    case_id: str
    level: CaseLevel
    cell_id: int
    t_s: float
    current_A: float
    voltage_true_V: float
    ocv_V: float
    soc_true: float
    v_rc_V: float
    capacity_Ah: float
    capacity_factor: float
    contact_resistance_factor: float
    r0_Ohm: float
    r1_Ohm: float
    c1_F: float
    fault_label: FaultLabel


@dataclass(frozen=True)
class SensedProfileRow:
    """Sensor-layer profile row after applying measurement deviations.

    Per methodology contract §6, v0.1 does not include BMS estimator
    output, such as ``reported_soc``, in this layer.

    Sensor behaviour and estimator behaviour are deliberately decoupled:

    - sensors apply voltage measurement bias;
    - estimator outputs live in diagnostics/reporting modules.

    Model-based estimators are deferred to later versions.
    """

    schema_version: str
    case_id: str
    level: CaseLevel
    cell_id: int
    t_s: float
    current_A: float
    voltage_true_V: float
    voltage_measured_V: float
    voltage_bias_mV: float
    ocv_V: float
    soc_true: float
    v_rc_V: float
    capacity_Ah: float
    capacity_factor: float
    contact_resistance_factor: float
    r0_Ohm: float
    r1_Ohm: float
    c1_F: float
    fault_label: FaultLabel


# ---------------------------------------------------------------------------
# Report rows: verdict layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticReportRow:
    """One diagnostic robustness report row per case.

    Each of the six v0.1 rules contributes one metric and one rule-level
    verdict. The aggregate ``final_verdict`` carries diagnostic meaning
    beyond simple PASS/WARNING/FAIL and therefore uses ``FinalVerdict``.

    Rule index:

    1. Coulomb-counting drift
       → ``soc_error_coulomb_max``, ``coulomb_drift_verdict``

    2. OCV-reset consistency
       → ``soc_error_ocv_reset_max``, ``ocv_reset_verdict``

    3. Voltage residual
       → ``voltage_residual_max_mV``, ``voltage_residual_verdict``

    4. Capacity consistency
       → ``capacity_consistency_error``, ``capacity_verdict``

    5. Contact-resistance proxy
       → ``contact_R_inferred_ratio``, ``contact_R_verdict``

    6. Module imbalance
       → ``module_delta_voltage_max_mV``, ``module_imbalance_verdict``

    ``capacity_consistency_error`` is defined as
    ``1 - C_inferred / C_nominal``.

    ``contact_R_inferred_ratio`` is defined as
    ``R0_inferred / R0_baseline``.
    """

    schema_version: str
    case_id: str
    fault_label: FaultLabel

    soc_error_coulomb_max: float
    soc_error_ocv_reset_max: float
    voltage_residual_max_mV: float
    capacity_consistency_error: float
    contact_R_inferred_ratio: float
    module_delta_voltage_max_mV: float

    coulomb_drift_verdict: RuleVerdict
    ocv_reset_verdict: RuleVerdict
    voltage_residual_verdict: RuleVerdict
    capacity_verdict: RuleVerdict
    contact_R_verdict: RuleVerdict
    module_imbalance_verdict: RuleVerdict

    false_positive: bool
    false_negative: bool
    final_verdict: FinalVerdict


@dataclass(frozen=True)
class ObservabilityRow:
    """Manually curated fault observability row.

    Each row describes the qualitative signature of one fault label
    across available measurement channels and the ambiguity level of that
    signature against the rest of the case matrix.

    Per methodology contract §8, this matrix is hand-written in v0.1.
    """

    schema_version: str
    fault_label: FaultLabel
    voltage_signature: str
    current_signature: str
    temperature_signature: str
    soc_signature: str
    soh_signature: str
    module_spread_signature: str
    ambiguity_level: str
    notes: str


@dataclass(frozen=True)
class ModuleInconsistencyReportRow:
    """One module-level inconsistency report row per case.

    Field naming is deliberately disambiguated:

    - ``pack_voltage_*_V``: extrema of pack-terminal voltage over time.
    - ``cell_voltage_*_anytime_V``: extrema across joint cell-time space.

    ``weakest_cell_id`` is reported together with
    ``weakest_cell_criterion`` so that "weakest" is explicitly defined.
    """

    schema_version: str
    case_id: str
    pack_voltage_max_V: float
    pack_voltage_min_V: float
    cell_voltage_max_anytime_V: float
    cell_voltage_min_anytime_V: float
    delta_cell_voltage_max_mV: float
    weakest_cell_id: int
    weakest_cell_criterion: WeakestCriterion
    module_risk_flag: RuleVerdict


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_case_spec(case: CaseSpec) -> None:
    """Validate a :class:`CaseSpec` against the v0.1 schema contract.

    Raises:
        ValueError: if any field violates the contract.
    """

    if not case.case_id:
        raise ValueError("case_id must not be empty")

    if case.level not in _VALID_LEVELS:
        raise ValueError(f"invalid level: {case.level!r}")

    if case.capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    if case.contact_resistance_factor <= 0:
        raise ValueError("contact_resistance_factor must be positive")

    if not isfinite(case.initial_soc_mismatch):
        raise ValueError("initial_soc_mismatch must be finite")

    if not isfinite(case.voltage_bias_mV):
        raise ValueError("voltage_bias_mV must be finite")

    if case.cell_scatter_type not in _VALID_CELL_SCATTER_TYPES:
        raise ValueError(f"invalid cell_scatter_type: {case.cell_scatter_type!r}")

    if not isfinite(case.cell_scatter_magnitude) or case.cell_scatter_magnitude < 0:
        raise ValueError("cell_scatter_magnitude must be finite and non-negative")

    if case.cell_scatter_type == "none" and case.cell_scatter_magnitude != 0.0:
        raise ValueError(
            "cell_scatter_type='none' requires cell_scatter_magnitude == 0.0"
        )

    if case.cell_scatter_type != "none" and case.cell_scatter_magnitude == 0.0:
        raise ValueError(
            "cell_scatter_type != 'none' requires cell_scatter_magnitude > 0"
        )

    if case.level == "cell" and case.cell_scatter_type != "none":
        raise ValueError(
            "cell-level case must have cell_scatter_type='none' "
            "(scatter is a module-level concept)"
        )

    if case.fault_label not in _VALID_FAULT_LABELS:
        raise ValueError(f"invalid fault_label: {case.fault_label!r}")

    if case.nominal_capacity_Ah <= 0:
        raise ValueError("nominal_capacity_Ah must be positive")

    if case.nominal_voltage_V <= 0:
        raise ValueError("nominal_voltage_V must be positive")

    if case.r0_baseline_Ohm <= 0:
        raise ValueError("r0_baseline_Ohm must be positive")

    if case.r1_baseline_Ohm <= 0:
        raise ValueError("r1_baseline_Ohm must be positive")

    if case.c1_baseline_F <= 0:
        raise ValueError("c1_baseline_F must be positive")


def validate_profile_row(row: ProfileRow) -> None:
    """Validate a ground-truth :class:`ProfileRow`."""

    _validate_common_profile_fields(
        schema_version=row.schema_version,
        level=row.level,
        cell_id=row.cell_id,
        t_s=row.t_s,
        current_A=row.current_A,
        voltage_true_V=row.voltage_true_V,
        ocv_V=row.ocv_V,
        soc_true=row.soc_true,
        v_rc_V=row.v_rc_V,
        capacity_Ah=row.capacity_Ah,
        capacity_factor=row.capacity_factor,
        contact_resistance_factor=row.contact_resistance_factor,
        r0_Ohm=row.r0_Ohm,
        r1_Ohm=row.r1_Ohm,
        c1_F=row.c1_F,
        fault_label=row.fault_label,
    )


def validate_sensed_profile_row(row: SensedProfileRow) -> None:
    """Validate a sensor-layer :class:`SensedProfileRow`.

    The v0.1 sensor model applies deterministic voltage bias only:

    ``voltage_measured_V = voltage_true_V + voltage_bias_mV / 1000``

    A small numerical tolerance is allowed for floating-point arithmetic.
    """

    _validate_common_profile_fields(
        schema_version=row.schema_version,
        level=row.level,
        cell_id=row.cell_id,
        t_s=row.t_s,
        current_A=row.current_A,
        voltage_true_V=row.voltage_true_V,
        ocv_V=row.ocv_V,
        soc_true=row.soc_true,
        v_rc_V=row.v_rc_V,
        capacity_Ah=row.capacity_Ah,
        capacity_factor=row.capacity_factor,
        contact_resistance_factor=row.contact_resistance_factor,
        r0_Ohm=row.r0_Ohm,
        r1_Ohm=row.r1_Ohm,
        c1_F=row.c1_F,
        fault_label=row.fault_label,
    )

    if not isfinite(row.voltage_measured_V):
        raise ValueError("voltage_measured_V must be finite")

    if not isfinite(row.voltage_bias_mV):
        raise ValueError("voltage_bias_mV must be finite")

    expected_measured = row.voltage_true_V + row.voltage_bias_mV / 1000.0
    if abs(row.voltage_measured_V - expected_measured) > 1e-9:
        raise ValueError(
            "voltage_measured_V must equal "
            "voltage_true_V + voltage_bias_mV / 1000"
        )


def _validate_common_profile_fields(
    *,
    schema_version: str,
    level: str,
    cell_id: int,
    t_s: float,
    current_A: float,
    voltage_true_V: float,
    ocv_V: float,
    soc_true: float,
    v_rc_V: float,
    capacity_Ah: float,
    capacity_factor: float,
    contact_resistance_factor: float,
    r0_Ohm: float,
    r1_Ohm: float,
    c1_F: float,
    fault_label: str,
) -> None:
    """Validate fields shared by truth-layer and sensor-layer profile rows."""

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"invalid schema_version: {schema_version!r} "
            f"(expected {SCHEMA_VERSION!r})"
        )

    if level not in _VALID_LEVELS:
        raise ValueError(f"invalid level: {level!r}")

    if cell_id < 1:
        raise ValueError("cell_id must be >= 1")

    if t_s < 0:
        raise ValueError("t_s must be non-negative")

    if not 0.0 <= soc_true <= 1.0:
        raise ValueError(f"soc_true must be within [0, 1], got {soc_true}")

    for name, value in (
        ("current_A", current_A),
        ("voltage_true_V", voltage_true_V),
        ("ocv_V", ocv_V),
        ("v_rc_V", v_rc_V),
        ("capacity_Ah", capacity_Ah),
        ("capacity_factor", capacity_factor),
        ("contact_resistance_factor", contact_resistance_factor),
        ("r0_Ohm", r0_Ohm),
        ("r1_Ohm", r1_Ohm),
        ("c1_F", c1_F),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")

    if capacity_Ah <= 0:
        raise ValueError("capacity_Ah must be positive")

    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    if contact_resistance_factor <= 0:
        raise ValueError("contact_resistance_factor must be positive")

    if r0_Ohm <= 0 or r1_Ohm <= 0 or c1_F <= 0:
        raise ValueError("ECM parameters must be positive")

    if fault_label not in _VALID_FAULT_LABELS:
        raise ValueError(f"invalid fault_label: {fault_label!r}")


def rows_to_dataframe(rows: Iterable[Any]) -> pd.DataFrame:
    """Convert dataclass rows into a :class:`pandas.DataFrame`.

    Each row must be a dataclass instance whose fields define the
    DataFrame columns. Field order in the dataclass definition is
    preserved as the column order of the resulting DataFrame.
    """

    return pd.DataFrame([asdict(row) for row in rows])


__all__ = [
    "SCHEMA_VERSION",
    "CaseLevel",
    "CellScatterType",
    "FaultLabel",
    "RuleVerdict",
    "FinalVerdict",
    "WeakestCriterion",
    "CaseSpec",
    "ProfileRow",
    "SensedProfileRow",
    "DiagnosticReportRow",
    "ObservabilityRow",
    "ModuleInconsistencyReportRow",
    "validate_case_spec",
    "validate_profile_row",
    "validate_sensed_profile_row",
    "rows_to_dataframe",
]
