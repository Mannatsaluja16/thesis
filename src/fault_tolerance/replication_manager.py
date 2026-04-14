import logging
import uuid
from dataclasses import dataclass
from src.config import REPLICATION_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReplicaInfo:
    replica_id: str
    task_id: str
    target_server: str
    active: bool = True


_replicas: dict = {}


def should_replicate(fault_confidence: float, task_priority: str) -> bool:
    """Replicate only when fault confidence > 0.75 AND task is critical."""
    return fault_confidence > REPLICATION_THRESHOLD and task_priority == "critical"


def create_replica(task_id: str, target_server: str) -> ReplicaInfo:
    replica_id = str(uuid.uuid4())
    replica = ReplicaInfo(replica_id=replica_id, task_id=task_id, target_server=target_server)
    _replicas[replica_id] = replica
    logger.info("Created replica %s for task %s on %s", replica_id, task_id, target_server)
    return replica


def remove_replica(replica_id: str) -> None:
    if replica_id in _replicas:
        _replicas[replica_id].active = False
        del _replicas[replica_id]
        logger.info("Removed replica %s", replica_id)
    else:
        logger.warning("Replica %s not found.", replica_id)


def main():
    logger.info("should_replicate(0.8, 'critical')   = %s", should_replicate(0.8, "critical"))
    logger.info("should_replicate(0.8, 'normal')     = %s", should_replicate(0.8, "normal"))
    logger.info("should_replicate(0.5, 'critical')   = %s", should_replicate(0.5, "critical"))

    replica = create_replica("task_42", "server_03")
    logger.info("Replica info: %s", replica)
    remove_replica(replica.replica_id)


if __name__ == "__main__":
    main()
