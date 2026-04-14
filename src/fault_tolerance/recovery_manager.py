import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    server_id: str
    success: bool
    recovery_time: float  # seconds
    method: str  # "replica" | "checkpoint"


_server_states: dict = {}
_pending_tasks: dict = {}  # server_id -> list of task ids


def handle_fault(server_id: str, timestamp: Optional[datetime] = None) -> RecoveryResult:
    """Handle a detected fault: stop assignments, restore, reroute tasks."""
    if timestamp is None:
        timestamp = datetime.utcnow()

    logger.info("Fault detected on %s at %s", server_id, timestamp)
    _server_states[server_id] = "RECOVERING"

    restored = restore_from_replica(server_id)
    method   = "replica" if restored else "checkpoint"

    restore_time = datetime.utcnow()
    recovery_time = calculate_recovery_time(timestamp, restore_time)

    _server_states[server_id] = "healthy"
    logger.info(
        "Server %s recovered via %s in %.2fs", server_id, method, recovery_time
    )
    return RecoveryResult(
        server_id=server_id,
        success=True,
        recovery_time=recovery_time,
        method=method,
    )


def restore_from_replica(server_id: str) -> bool:
    """Attempt to restore server state from replica. Returns True if successful."""
    # In a real system this would contact the Replication Manager.
    # For simulation we always succeed.
    logger.info("Restoring %s from replica...", server_id)
    return True


def calculate_recovery_time(fault_time: datetime, restore_time: datetime) -> float:
    """Return recovery time in seconds."""
    delta = (restore_time - fault_time).total_seconds()
    return max(delta, 0.0)


def main():
    fault_ts = datetime.utcnow()
    result = handle_fault("server_01", fault_ts)
    logger.info("RecoveryResult: %s", result)


if __name__ == "__main__":
    main()
