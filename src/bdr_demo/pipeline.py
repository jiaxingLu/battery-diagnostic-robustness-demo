"""End-to-end v0.1 report pipeline.

The pipeline runs all locked v0.1 diagnostic cases and exports:

- diagnostic_robustness_report.csv
- module_inconsistency_report.csv
- observability_matrix.csv
- diagnostic plots

This is the first executable end-to-end layer of the demonstrator:

    CaseSpec
    → synthetic 1RC truth profiles
    → 24s1p series-module aggregation
    → deterministic sensor layer
    → rule-based diagnostics
    → CSV reports and plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from bdr_demo.case_matrix import V0_1_CASES
from bdr_demo.diagnostics import compute_diagnostic_report
from bdr_demo.module_aggregator import (
    aggregate_series_module,
    generate_series_module_cell_profiles,
    make_module_inconsistency_report,
)
from bdr_demo.observability import export_observability_matrix
from bdr_demo.schema import (
    CaseSpec,
    DiagnosticReportRow,
    ModuleInconsistencyReportRow,
    rows_to_dataframe,
)
from bdr_demo.sensors import apply_case_sensor_model


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = _REPO_ROOT / "reports"
DEFAULT_PLOT_DIR = DEFAULT_REPORT_DIR / "plots"


def run_case(
    case: CaseSpec,
    *,
    initial_soc_assumed: float = 0.80,
    discharge_current_A: float = 1.0,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
) -> tuple[DiagnosticReportRow, ModuleInconsistencyReportRow, pd.DataFrame]:
    """Run one v0.1 case through the full report pipeline.

    Returns:
        diagnostic report row, module inconsistency report row, module DataFrame.
    """

    truth_rows = generate_series_module_cell_profiles(
        case,
        initial_soc_assumed=initial_soc_assumed,
        discharge_current_A=discharge_current_A,
        t_end_s=t_end_s,
        dt_s=dt_s,
    )

    module_df = aggregate_series_module(truth_rows)
    sensed_rows = apply_case_sensor_model(case, truth_rows)

    diagnostic_row = compute_diagnostic_report(
        case,
        sensed_rows,
        module_df=module_df,
        initial_soc_assumed=initial_soc_assumed,
    )

    module_row = make_module_inconsistency_report(truth_rows, module_df)

    return diagnostic_row, module_row, module_df


def run_all_cases(
    cases: Iterable[CaseSpec] = V0_1_CASES,
    *,
    initial_soc_assumed: float = 0.80,
    discharge_current_A: float = 1.0,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all requested cases and return diagnostic and module reports."""

    diagnostic_rows: list[DiagnosticReportRow] = []
    module_rows: list[ModuleInconsistencyReportRow] = []

    for case in cases:
        diagnostic_row, module_row, _ = run_case(
            case,
            initial_soc_assumed=initial_soc_assumed,
            discharge_current_A=discharge_current_A,
            t_end_s=t_end_s,
            dt_s=dt_s,
        )
        diagnostic_rows.append(diagnostic_row)
        module_rows.append(module_row)

    diagnostic_df = rows_to_dataframe(diagnostic_rows)
    module_df = rows_to_dataframe(module_rows)

    return diagnostic_df, module_df


def _plot_final_verdict_counts(diagnostic_df: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot count of final diagnostic verdicts."""

    import matplotlib.pyplot as plt

    counts = diagnostic_df["final_verdict"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Final diagnostic verdict counts")
    ax.set_xlabel("Final verdict")
    ax.set_ylabel("Number of cases")
    ax.grid(axis="y", alpha=0.35)

    fig.tight_layout()
    out = plot_dir / "diagnostic_verdict_counts.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    return out


def _plot_rule_verdict_matrix(diagnostic_df: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot rule-level verdicts as a compact matrix."""

    import matplotlib.pyplot as plt
    import numpy as np

    rule_cols = [
        "coulomb_drift_verdict",
        "ocv_reset_verdict",
        "voltage_residual_verdict",
        "capacity_verdict",
        "contact_R_verdict",
        "module_imbalance_verdict",
    ]
    mapping = {"PASS": 0, "WARNING": 1, "FAIL": 2}

    matrix = (
        diagnostic_df[rule_cols]
        .apply(lambda col: col.map(mapping))
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=2)

    ax.set_title("Rule-level diagnostic verdict matrix")
    ax.set_yticks(np.arange(len(diagnostic_df)))
    ax.set_yticklabels(diagnostic_df["case_id"])
    ax.set_xticks(np.arange(len(rule_cols)))
    ax.set_xticklabels(
        [
            "Coulomb",
            "OCV reset",
            "Voltage residual",
            "Capacity",
            "Contact-R",
            "Module spread",
        ],
        rotation=35,
        ha="right",
    )

    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["PASS", "WARNING", "FAIL"])

    fig.tight_layout()
    out = plot_dir / "rule_verdict_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    return out


def _plot_key_metrics(diagnostic_df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    """Plot selected diagnostic metrics as separate bar charts."""

    import matplotlib.pyplot as plt

    metric_specs = [
        ("soc_error_coulomb_max", "Max Coulomb-counting SOC error [-]"),
        ("soc_error_ocv_reset_max", "Max OCV-reset SOC error [-]"),
        ("voltage_residual_max_mV", "Max voltage residual [mV]"),
        ("capacity_consistency_error", "Capacity consistency error [-]"),
        ("contact_R_inferred_ratio", "Contact-R inferred ratio [-]"),
        ("module_delta_voltage_max_mV", "Max module voltage spread [mV]"),
    ]

    paths: list[Path] = []

    for col, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(diagnostic_df["case_id"], diagnostic_df[col])
        ax.set_title(col)
        ax.set_xlabel("Case")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.35)
        ax.tick_params(axis="x", rotation=35)

        fig.tight_layout()
        out = plot_dir / f"{col}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)

    return paths


def create_report_plots(diagnostic_df: pd.DataFrame, plot_dir: str | Path) -> list[Path]:
    """Create formal report plots from diagnostic report data."""

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_final_verdict_counts(diagnostic_df, out_dir),
        _plot_rule_verdict_matrix(diagnostic_df, out_dir),
    ]
    paths.extend(_plot_key_metrics(diagnostic_df, out_dir))

    for path in paths:
        print(f"[OK] saved plot: {path}")

    return paths


def run_pipeline(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    make_plots: bool = True,
    initial_soc_assumed: float = 0.80,
    discharge_current_A: float = 1.0,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full v0.1 report pipeline and write reports to disk."""

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_df, module_df = run_all_cases(
        initial_soc_assumed=initial_soc_assumed,
        discharge_current_A=discharge_current_A,
        t_end_s=t_end_s,
        dt_s=dt_s,
    )

    diagnostic_path = out_dir / "diagnostic_robustness_report.csv"
    module_path = out_dir / "module_inconsistency_report.csv"
    observability_path = out_dir / "observability_matrix.csv"

    diagnostic_df.to_csv(diagnostic_path, index=False)
    module_df.to_csv(module_path, index=False)
    export_observability_matrix(observability_path)

    print(f"[OK] saved diagnostic report: {diagnostic_path}")
    print(f"[OK] saved module report: {module_path}")

    if make_plots:
        create_report_plots(diagnostic_df, out_dir / "plots")

    return diagnostic_df, module_df


def main() -> None:
    """CLI entry point for manual execution."""

    diagnostic_df, module_df = run_pipeline()

    print("\n[diagnostic report]")
    print(
        diagnostic_df[
            [
                "case_id",
                "fault_label",
                "coulomb_drift_verdict",
                "ocv_reset_verdict",
                "voltage_residual_verdict",
                "capacity_verdict",
                "contact_R_verdict",
                "module_imbalance_verdict",
                "final_verdict",
            ]
        ].to_string(index=False)
    )

    print("\n[module report]")
    print(
        module_df[
            [
                "case_id",
                "delta_cell_voltage_max_mV",
                "weakest_cell_id",
                "module_risk_flag",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
