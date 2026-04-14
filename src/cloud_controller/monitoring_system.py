import csv
import logging
import os
import threading
import time
from datetime import datetime

import numpy as np

from src.cloud_controller.resource_manager import (
    get_all_servers, update_server_state,
)
from src.fault_tolerance.vm_migration import (
    is_migration_needed, migrate_vm, select_migration_target,
)
from src.fault_tolerance.replication_manager import should_replicate, create_replica
from src.fault_tolerance.recovery_manager import handle_fault
from src.config import WINDOW_SIZE, INPUT_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions: list = []
_lock = threading.Lock()
_running = False
_predictor = None

EVENTS_CSV = "results/reports/events.csv"


def _get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from src.fault_prediction.predict import predict_fault
            _predictor = predict_fault
            logger.info("Fault predictor loaded.")
        except Exception as e:
            logger.warning("Could not load predictor (%s). Using dummy.", e)
            _predictor = lambda w, instance_id="": {
                "fault_predicted": False, "confidence": 0.0, "instance_id": instance_id
            }
    return _predictor


def _log_event(event_type: str, details: str):
    os.makedirs("results/reports", exist_ok=True)
    write_header = not os.path.exists(EVENTS_CSV)
    with open(EVENTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "event_type", "details"])
        writer.writerow([datetime.utcnow().isoformat(), event_type, details])


_replay = None  # type: np.ndarray or None  — (500, 10, 5) real Kaggle test sequences
_replay_cursor: int = 0            # next sequence index to assign


def _load_replay():
    """Load pre-sampled real test sequences once."""
    global _replay
    if _replay is None:
        replay_path = os.path.join(os.path.dirname(__file__), "monitoring_replay.npy")
        try:
            _replay = np.load(replay_path)
            logger.info("Loaded %d real monitoring sequences from %s", len(_replay), replay_path)
        except FileNotFoundError:
            logger.warning("monitoring_replay.npy not found — falling back to synthetic windows")
            _replay = np.array([])
    return _replay


def _next_window() -> np.ndarray:
    """Return the next real sequence from the replay buffer (cycles endlessly)."""
    global _replay_cursor
    replay = _load_replay()
    if replay.size == 0:
        return np.random.rand(WINDOW_SIZE, INPUT_SIZE).astype(np.float32)
    window = replay[_replay_cursor % len(replay)]
    _replay_cursor += 1
    return window.astype(np.float32)


def _poll_once():
    """Single monitoring cycle: update metrics → predict → act."""
    predict_fault = _get_predictor()

    servers = get_all_servers()

    # Recover QUARANTINED servers back to healthy each cycle so the active
    # pool never drains and faults keep appearing in subsequent cycles.
    for server_id, state in servers.items():
        if state["status"] == "QUARANTINED":
            update_server_state(server_id, {"status": "healthy"})
            _log_event("RECOVERED", f"server={server_id} auto-recovered to healthy")
            logger.info("Server %s auto-recovered to healthy", server_id)

    # Re-fetch after recovery so we process the full fleet
    servers = get_all_servers()

    for server_id, state in servers.items():
        if state["status"] not in ("healthy", "at-risk"):
            continue

        # Use a real Kaggle test sequence for this server so the model produces
        # accurate, meaningful confidence scores (not synthetic windows).
        window = _next_window()

        # Reflect the sequence's normalised CPU back as a display metric (feature 0).
        cpu_display = round(float(window[:, 0].mean()) * 100, 1)
        mem_display = round(float(window[:, 4].mean()) * 100, 1)  # disk_w ≈ mem proxy
        update_server_state(server_id, {"cpu": max(cpu_display, 1.0),
                                        "mem": max(mem_display, 1.0)})

        prediction = predict_fault(window, instance_id=server_id)

        with _lock:
            _predictions.append(prediction)
            if len(_predictions) > 500:
                _predictions.pop(0)

        if prediction["fault_predicted"]:
            update_server_state(server_id, {"status": "at-risk"})
            _log_event("FAULT_PREDICTED", f"server={server_id} confidence={prediction['confidence']:.3f}")
            logger.warning("Fault predicted for %s (confidence=%.3f)", server_id, prediction["confidence"])

            # Trigger migration
            healthy = [s for s in servers.values() if s["status"] == "healthy"]
            if healthy:
                try:
                    target = select_migration_target(list(servers.values()), exclude=server_id)
                    result = migrate_vm(server_id, target, state.get("vms", []))
                    update_server_state(server_id, {"status": "QUARANTINED"})
                    _log_event("VM_MIGRATED", f"source={server_id} target={target}")
                except ValueError:
                    logger.error("No migration target for %s", server_id)

        # Check for actual fault (StatusCheckFailed)
        if state.get("StatusCheckFailed", 0):
            result = handle_fault(server_id, datetime.utcnow())
            _log_event("RECOVERY", f"server={server_id} recovery_time={result.recovery_time:.2f}s")


def start_monitoring(interval_seconds: int = 60):
    """Start background monitoring thread."""
    global _running
    _running = True

    def _loop():
        while _running:
            try:
                _poll_once()
            except Exception as e:
                logger.error("Monitoring error: %s", e)
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True, name="MonitoringThread")
    t.start()
    logger.info("Monitoring started (interval=%ds)", interval_seconds)
    return t


def stop_monitoring():
    global _running
    _running = False


def get_latest_predictions() -> list:
    with _lock:
        return list(_predictions[-50:])


def main():
    _poll_once()
    logger.info("Latest predictions: %s", get_latest_predictions())


if __name__ == "__main__":
    main()
