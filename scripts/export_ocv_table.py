"""Export a static OCV-SOC table from the PyBaMM Chen2020 parameter set.

This script is a build-time helper only. It requires PyBaMM but the v0.1
runtime package does not.

Method:
    For each SOC grid point, initialize a PyBaMM SPM at that SOC, run a
    short zero-current rest step, and record the terminal voltage.

This avoids manually composing positive/negative electrode OCP functions
and lets PyBaMM handle the internal stoichiometry mapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_PATH = Path("data/ocv/ocv_chen2020.csv")
N_POINTS = 201
PARAMETER_SET = "Chen2020"
MODEL_NAME = "SPM"
EXPORT_METHOD = "pybamm_model_initial_soc_rest_voltage"


def export_ocv_table() -> pd.DataFrame:
    """Build and return the OCV-SOC table."""

    try:
        import pybamm
    except ImportError as exc:
        raise SystemExit(
            "PyBaMM is required only for this build-time export script.\n"
            'Install build dependencies with: python3 -m pip install -e ".[build]"'
        ) from exc

    soc_grid = np.linspace(0.0, 1.0, N_POINTS)
    rows: list[dict[str, object]] = []

    for soc in soc_grid:
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues(PARAMETER_SET)

        # A one-second rest is enough because no load current is applied.
        # The goal is to read the initialized equilibrium-like terminal voltage.
        experiment = pybamm.Experiment(["Rest for 1 second"], period="1 second")

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )

        try:
            sol = sim.solve(initial_soc=float(soc))
        except Exception as exc:
            raise RuntimeError(
                f"PyBaMM failed while exporting OCV at SOC={soc:.5f}."
            ) from exc

        voltage = float(sol["Terminal voltage [V]"](sol.t[-1]))

        rows.append(
            {
                "soc": float(soc),
                "ocv_V": voltage,
                "source": PARAMETER_SET,
                "model": MODEL_NAME,
                "export_method": EXPORT_METHOD,
            }
        )

    df = pd.DataFrame(rows)

    if not df["soc"].is_monotonic_increasing:
        raise RuntimeError("SOC grid is not monotonic increasing.")

    if df["ocv_V"].isna().any():
        raise RuntimeError("OCV table contains NaN values.")

    if not ((df["ocv_V"] > 2.0) & (df["ocv_V"] < 5.0)).all():
        bad = df.loc[~((df["ocv_V"] > 2.0) & (df["ocv_V"] < 5.0))]
        raise RuntimeError(f"OCV values outside plausible bounds:\n{bad}")

    return df


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = export_ocv_table()
    df.to_csv(OUT_PATH, index=False)

    print(f"[OK] Saved OCV table: {OUT_PATH}")
    print(f"[rows] {len(df)}")
    print(f"[soc] {df['soc'].min():.3f} → {df['soc'].max():.3f}")
    print(f"[ocv_V] {df['ocv_V'].min():.4f} → {df['ocv_V'].max():.4f}")


if __name__ == "__main__":
    main()
