from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("reports/debug")
OUTPUT_DIR = INPUT_DIR / "plots"

FILES = [
    "baseline_profile.csv",
    "contact_resistance_150_profile.csv",
    "capacity_fade_90_profile.csv",
    "initial_soc_mismatch_p05_profile.csv",
]


def load_profiles():
    profiles = {}
    for fname in FILES:
        path = INPUT_DIR / fname
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        profiles[path.stem] = pd.read_csv(path)
    return profiles


def plot_voltage(profiles):
    plt.figure(figsize=(8, 5))
    for name, df in profiles.items():
        plt.plot(df["t_s"], df["voltage_true_V"], label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("Terminal voltage [V]")
    plt.title("Voltage profiles")
    plt.legend()
    plt.grid(True)
    out = OUTPUT_DIR / "voltage_profiles.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] saved {out}")


def plot_soc(profiles):
    plt.figure(figsize=(8, 5))
    for name, df in profiles.items():
        plt.plot(df["t_s"], df["soc_true"], label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("SOC [-]")
    plt.title("SOC profiles")
    plt.legend()
    plt.grid(True)
    out = OUTPUT_DIR / "soc_profiles.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] saved {out}")


def plot_components(profiles):
    for name, df in profiles.items():
        plt.figure(figsize=(8, 5))
        plt.plot(df["t_s"], df["ocv_V"], label="ocv_V")
        plt.plot(df["t_s"], df["v_rc_V"], label="v_rc_V")
        plt.plot(df["t_s"], df["voltage_true_V"], label="voltage_true_V")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Voltage components — {name}")
        plt.legend()
        plt.grid(True)
        out = OUTPUT_DIR / f"{name}_components.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] saved {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    if not profiles:
        raise RuntimeError("No debug profile CSV files found.")

    plot_voltage(profiles)
    plot_soc(profiles)
    plot_components(profiles)

    print("\n[done] plots written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()