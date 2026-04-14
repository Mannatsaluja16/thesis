"""
real_data_loader.py
Fault injection on real VM workload traces.
Uses real data as normal baseline + injects realistic fault events (ramp + sustained fault + recovery)
scaled to each VM own distribution. Target ~13% fault rows.
"""
import glob, logging, os, random
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(42); np.random.seed(42)

FAULT_RATE   = 0.13
RAMP_LEN     = 5        # rows of gradually escalating metrics
FAULT_LEN    = 2        # rows of peak (StatusCheckFailed=1)
RECOVERY_LEN = 2        # rows recovering
EVENT_LEN    = RAMP_LEN + FAULT_LEN + RECOVERY_LEN   # 9 rows per event
# fault_rate = FAULT_LEN / (GROUP_SIZE + EVENT_LEN)  →  GROUP_SIZE = FAULT_LEN/FAULT_RATE - EVENT_LEN
GROUP_SIZE   = max(2, int(FAULT_LEN / FAULT_RATE) - EVENT_LEN)  # ~6 normal rows between events

DATASET_DIR  = "dataset"
RAW_OUTPUT   = "data/raw/real_workload.csv"
FEATURE_COLS = ["CPUUtilization","NetworkIn","NetworkOut","DiskReadOps","DiskWriteOps"]


def _inject_faults(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n  = len(df)
    i  = GROUP_SIZE   # start after first normal block
    while i + EVENT_LEN < n:
        # add jitter so events are not perfectly periodic
        jitter = random.randint(-5, 5)
        start  = max(0, i + jitter)
        if start + EVENT_LEN >= n:
            break
        # Ramp rows
        for step in range(RAMP_LEN):
            p   = (step + 1) / RAMP_LEN
            idx = start + step
            df.at[idx, "CPUUtilization"] = float(np.clip(s["cpu_m"] + p * 3.5 * s["cpu_s"] + np.random.normal(0, s["cpu_s"]*0.3), 0, 100))
            df.at[idx, "DiskWriteOps"]   = float(np.clip(s["dw_m"]  + p * 3.5 * s["dw_s"]  + np.random.normal(0, s["dw_s"]*0.3),  0, None))
            df.at[idx, "NetworkIn"]      = float(np.clip(s["ni_m"]  + p * 3.0 * s["ni_s"]  + np.random.normal(0, s["ni_s"]*0.3),  0, None))
            df.at[idx, "StatusCheckFailed"] = 0
        # Fault rows
        for step in range(FAULT_LEN):
            idx = start + RAMP_LEN + step
            df.at[idx, "CPUUtilization"] = float(np.clip(s["cpu_m"] + 4.5 * s["cpu_s"] + np.random.normal(0, s["cpu_s"]*0.2), 0, 100))
            df.at[idx, "DiskWriteOps"]   = float(np.clip(s["dw_m"]  + 4.5 * s["dw_s"]  + np.random.normal(0, s["dw_s"]*0.2),  0, None))
            df.at[idx, "NetworkIn"]      = float(np.clip(s["ni_m"]  + 4.0 * s["ni_s"]  + np.random.normal(0, s["ni_s"]*0.2),  0, None))
            df.at[idx, "StatusCheckFailed"] = 1
        # Recovery rows
        for step in range(RECOVERY_LEN):
            decay = (step + 1) / (RECOVERY_LEN + 1)
            idx   = start + RAMP_LEN + FAULT_LEN + step
            df.at[idx, "CPUUtilization"] = float(np.clip(s["cpu_m"] * (1 + (1-decay)) + np.random.normal(0, s["cpu_s"]*0.2), 0, 100))
            df.at[idx, "StatusCheckFailed"] = 0
        i = start + EVENT_LEN + GROUP_SIZE
    return df


def load_single_vm(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    iid = "vm-" + os.path.splitext(os.path.basename(filepath))[0]
    df = df.rename(columns={
        "CPU usage [%]":                        "CPUUtilization",
        "Network received throughput [KB/s]":   "NetworkIn",
        "Network transmitted throughput [KB/s]":"NetworkOut",
        "Disk read throughput [KB/s]":          "DiskReadOps",
        "Disk write throughput [KB/s]":         "DiskWriteOps",
    })
    df["timestamp"]         = pd.to_datetime(df["Timestamp [ms]"], unit="s")
    df["instance_id"]       = iid
    df["StatusCheckFailed"] = 0
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0).clip(lower=0).astype(float)
    df = df[["timestamp","instance_id"] + FEATURE_COLS + ["StatusCheckFailed"]].copy()
    s = {
        "cpu_m": df["CPUUtilization"].mean(), "cpu_s": max(df["CPUUtilization"].std(), 0.5),
        "ni_m":  df["NetworkIn"].mean(),      "ni_s":  max(df["NetworkIn"].std(), 0.5),
        "dw_m":  df["DiskWriteOps"].mean(),   "dw_s":  max(df["DiskWriteOps"].std(), 0.5),
    }
    return _inject_faults(df, s)


def load_all_vms(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir!r}")
    frames = []
    for f in files:
        vdf = load_single_vm(f)
        n   = int(vdf["StatusCheckFailed"].sum())
        frames.append(vdf)
        logger.info("%-12s  rows=%5d  faults=%4d  (%.1f%%)", os.path.basename(f), len(vdf), n, n/len(vdf)*100)
    combined = pd.concat(frames, ignore_index=True).sort_values(["instance_id","timestamp"]).reset_index(drop=True)
    total = int(combined["StatusCheckFailed"].sum())
    logger.info("Combined: %d rows | %d faults (%.1f%%)", len(combined), total, total/len(combined)*100)
    return combined


def main():
    os.makedirs("data/raw", exist_ok=True)
    df = load_all_vms(DATASET_DIR)
    df.to_csv(RAW_OUTPUT, index=False)
    logger.info("Saved to %s", RAW_OUTPUT)

if __name__ == "__main__":
    main()
