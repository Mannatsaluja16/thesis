import logging
from dataclasses import dataclass, field
from typing import List
from src.config import FAULT_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    success: bool
    source_id: str
    target_id: str
    workloads: List[str]
    migration_time: float  # seconds


def select_migration_target(servers: list, exclude: str) -> str:
    """Select the healthiest server (lowest CPU + memory load), excluding the source."""
    candidates = [s for s in servers if s["server_id"] != exclude and s["status"] == "healthy"]
    if not candidates:
        raise ValueError("No healthy migration target available.")
    return min(candidates, key=lambda s: s["cpu"] + s.get("mem", 0))["server_id"]


def migrate_vm(source_id: str, target_id: str, workloads: list) -> MigrationResult:
    """Migrate workloads from source to target server."""
    import time
    migration_time = len(workloads) * 0.5  # 0.5s per workload (simulation)
    logger.info(
        "Migrating %d workloads from %s → %s (estimated %.1fs)",
        len(workloads), source_id, target_id, migration_time,
    )
    return MigrationResult(
        success=True,
        source_id=source_id,
        target_id=target_id,
        workloads=workloads,
        migration_time=migration_time,
    )


def is_migration_needed(prediction: dict) -> bool:
    return prediction.get("fault_predicted", False)


def main():
    servers = [
        {"server_id": "server_01", "cpu": 80.0, "mem": 70.0, "status": "healthy"},
        {"server_id": "server_02", "cpu": 30.0, "mem": 40.0, "status": "healthy"},
        {"server_id": "server_03", "cpu": 20.0, "mem": 25.0, "status": "healthy"},
    ]
    target = select_migration_target(servers, exclude="server_01")
    logger.info("Selected migration target: %s", target)
    result = migrate_vm("server_01", target, ["task_1", "task_2"])
    logger.info("Migration result: %s", result)


if __name__ == "__main__":
    main()
