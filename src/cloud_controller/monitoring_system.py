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


def _poll_once():
    """Single monitoring cycle: poll metrics → predict → act."""
    predict_fault = _get_predictor()
    servers = get_all_servers()

    for server_id, state in servers.items():
        if state["status"] not in ("healthy", "at-risk"):
            continue

        # Build a synthetic 10-step window from current metrics
        cpu  = state.get("cpu", 40) / 100.0
        mem  = state.get("mem", 40) / 100.0
        window = np.tile([cpu, mem * 2e5, mem * 1.5e5, 30, 20], (WINDOW_SIZE, 1)).astype(np.float32)

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
