"""Debug plotting for generated single-cell v0.1 profiles.

This script reads profile CSV files from reports/debug/*_profile.csv and
writes PNG figures to reports/debug/plots/.

It is a development/debug helper, not part of the core v0.1 runtime pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = Path("reports/debug")
OUTPUT_DIR = INPUT_DIR / "plots"
SUMMARY_PATH = OUTPUT_DIR / "debug_profile_summary.csv"


def load_profiles(input_dir: Path = INPUT_DIR) -> dict[str, pd.DataFrame]:
    profiles: dict[str, pd.DataFrame] = {}

    for path in sorted(input_dir.glob("*_profile.csv")):
        df = pd.read_csv(path)
        required = {
            "t_s",
            "current_A",
            "soc_true",
            "ocv_V",
            "v_rc_V",
            "voltage_true_V",
            "r0_Ohm",
            "capacity_Ah",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        profiles[path.stem] = df

    if not profiles:
        raise RuntimeError(f"No *_profile.csv files found in {input_dir}")

    return profiles


def summarize_profiles(profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in profiles.items():
        rows.append(
            {
                "profile": name,
                "n_rows": len(df),
                "t_start_s": df["t_s"].iloc[0],
                "t_end_s": df["t_s"].iloc[-1],
                "soc_start": df["soc_true"].iloc[0],
                "soc_end": df["soc_true"].iloc[-1],
                "soc_drop": df["soc_true"].iloc[0] - df["soc_true"].iloc[-1],
                "voltage_start_V": df["voltage_true_V"].iloc[0],
                "voltage_end_V": df["voltage_true_V"].iloc[-1],
                "voltage_drop_V": df["voltage_true_V"].iloc[0]
                - df["voltage_true_V"].iloc[-1],
                "ocv_start_V": df["ocv_V"].iloc[0],
                "ocv_end_V": df["ocv_V"].iloc[-1],
                "v_rc_end_mV": 1000.0 * df["v_rc_V"].iloc[-1],
                "r0_Ohm": df["r0_Ohm"].iloc[0],
                "capacity_Ah": df["capacity_Ah"].iloc[0],
            }
        )

    return pd.DataFrame(rows)


def plot_voltage_profiles(profiles: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(9, 5.5))

    for name, df in profiles.items():
        plt.plot(df["t_s"], df["voltage_true_V"], linewidth=1.8, label=name)

    plt.xlabel("Time [s]")
    plt.ylabel("Terminal voltage [V]")
    plt.title("Terminal-voltage profiles")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "voltage_profiles.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] saved {out}")


def plot_voltage_delta_vs_baseline(profiles: dict[str, pd.DataFrame]) -> None:
    if "baseline_profile" not in profiles:
        print("[skip] baseline_profile not found; cannot plot voltage delta.")
        return

    baseline = profiles["baseline_profile"].set_index("t_s")

    plt.figure(figsize=(9, 5.5))

    for name, df in profiles.items():
        if name == "baseline_profile":
            continue

        aligned = df.set_index("t_s").join(
            baseline[["voltage_true_V"]].rename(
                columns={"voltage_true_V": "voltage_baseline_V"}
            ),
            how="inner",
        )

        delta_mV = 1000.0 * (
            aligned["voltage_true_V"] - aligned["voltage_baseline_V"]
        )
        plt.plot(aligned.index, delta_mV, linewidth=1.8, label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage deviation from baseline [mV]")
    plt.title("Terminal-voltage deviation relative to baseline")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "voltage_delta_vs_baseline.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] saved {out}")


def plot_soc_profiles(profiles: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(9, 5.5))

    for name, df in profiles.items():
        plt.plot(df["t_s"], df["soc_true"], linewidth=1.8, label=name)

    plt.xlabel("Time [s]")
    plt.ylabel("SOC [-]")
    plt.title("SOC profiles")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "soc_profiles.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] saved {out}")


def plot_soc_delta_vs_baseline(profiles: dict[str, pd.DataFrame]) -> None:
    if "baseline_profile" not in profiles:
        print("[skip] baseline_profile not found; cannot plot SOC delta.")
        return

    baseline = profiles["baseline_profile"].set_index("t_s")

    plt.figure(figsize=(9, 5.5))

    for name, df in profiles.items():
        if name == "baseline_profile":
            continue

        aligned = df.set_index("t_s").join(
            baseline[["soc_true"]].rename(columns={"soc_true": "soc_baseline"}),
            how="inner",
        )

        delta_soc_pct = 100.0 * (aligned["soc_true"] - aligned["soc_baseline"])
        plt.plot(aligned.index, delta_soc_pct, linewidth=1.8, label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("SOC deviation from baseline [percentage points]")
    plt.title("SOC deviation relative to baseline")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "soc_delta_vs_baseline.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] saved {out}")


def plot_components_per_profile(profiles: dict[str, pd.DataFrame]) -> None:
    for name, df in profiles.items():
        fig, ax1 = plt.subplots(figsize=(9, 5.5))

        ax1.plot(df["t_s"], df["ocv_V"], linewidth=1.8, label="OCV [V]")
        ax1.plot(
            df["t_s"],
            df["voltage_true_V"],
            linewidth=1.8,
            label="Terminal voltage [V]",
        )
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [V]")
        ax1.grid(True, alpha=0.35)

        ax2 = ax1.twinx()
        ax2.plot(
            df["t_s"],
            1000.0 * df["v_rc_V"],
            linestyle="--",
            linewidth=1.6,
            label="RC polarization [mV]",
        )
        ax2.set_ylabel("RC polarization [mV]")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=9)

        plt.title(f"Voltage components — {name}")
        fig.tight_layout()

        out = OUTPUT_DIR / f"{name}_components.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"[OK] saved {out}")


def plot_rc_polarization_all(profiles: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(9, 5.5))

    for name, df in profiles.items():
        plt.plot(df["t_s"], 1000.0 * df["v_rc_V"], linewidth=1.8, label=name)

    plt.xlabel("Time [s]")
    plt.ylabel("RC polarization [mV]")
    plt.title("RC polarization profiles")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "rc_polarization_profiles.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] saved {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles()
    summary = summarize_profiles(profiles)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("[summary]")
    print(summary.to_string(index=False))
    print(f"[OK] saved {SUMMARY_PATH}")

    plot_voltage_profiles(profiles)
    plot_voltage_delta_vs_baseline(profiles)
    plot_soc_profiles(profiles)
    plot_soc_delta_vs_baseline(profiles)
    plot_components_per_profile(profiles)
    plot_rc_polarization_all(profiles)

    print(f"\n[done] plots written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
