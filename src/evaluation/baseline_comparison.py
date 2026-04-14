import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ["baseline", "reactive", "proposed"]


def run_experiment(mode: str, n_runs: int = 5) -> dict:
    """Import and run the datacenter simulation for the requested mode."""
    from simulation.datacenter_sim import run_experiment as sim_run
    results = sim_run(mode, n_runs=n_runs)
    df = pd.DataFrame(results)
    numeric = df.select_dtypes(include=np.number)
    return {
        "mode":   mode,
        "mean":   numeric.mean().to_dict(),
        "std":    numeric.std().to_dict(),
        "raw":    results,
    }


def aggregate_results(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"mode": r["mode"]}
        row.update({f"{k}_mean": v for k, v in r["mean"].items()})
        row.update({f"{k}_std":  v for k, v in r["std"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: str = "results/plots") -> None:
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [
        ("energy_wh_mean",       "Energy Consumption (Wh)"),
        ("reliability_pct_mean", "Reliability (%)"),
        ("avg_recovery_s_mean",  "Avg Recovery Time (s)"),
        ("throughput_mean",      "Throughput (tasks/step)"),
        ("cost_usd_mean",        "Cost (USD)"),
    ]

    for col, title in metrics_to_plot:
        if col not in df.columns:
            continue
        std_col = col.replace("_mean", "_std")
        fig, ax = plt.subplots(figsize=(7, 4))
        modes = df["mode"].tolist()
        means = df[col].tolist()
        stds  = df[std_col].tolist() if std_col in df.columns else [0] * len(means)
        ax.bar(modes, means, yerr=stds, capsize=6, color=["#e74c3c", "#f39c12", "#2ecc71"])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("Mode")
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{col}.png")
        plt.savefig(fname)
        plt.close()
        logger.info("Saved plot: %s", fname)


def main():
    all_results = []
    for mode in MODES:
        logger.info("Running experiment: mode=%s", mode)
        r = run_experiment(mode, n_runs=5)
        all_results.append(r)

    df = aggregate_results(all_results)
    os.makedirs("results/reports", exist_ok=True)
    df.to_csv("results/reports/baseline_comparison.csv", index=False)
    logger.info("Saved comparison to results/reports/baseline_comparison.csv")
    print(df.to_string(index=False))
    plot_comparison(df)


if __name__ == "__main__":
    main()
