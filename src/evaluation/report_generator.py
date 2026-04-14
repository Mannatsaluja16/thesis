import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.baseline_comparison import run_experiment, aggregate_results, plot_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ["baseline", "reactive", "proposed"]


def main():
    all_results = []
    for mode in MODES:
        logger.info("Running experiment: mode=%s", mode)
        r = run_experiment(mode, n_runs=5)
        all_results.append(r)

    df = aggregate_results(all_results)

    os.makedirs("results/reports", exist_ok=True)
    df.to_csv("results/reports/final_results.csv", index=False)
    logger.info("Final results saved to results/reports/final_results.csv")

    os.makedirs("results/plots", exist_ok=True)
    plot_comparison(df, output_dir="results/plots")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON REPORT")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
