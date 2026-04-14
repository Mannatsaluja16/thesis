import itertools
import logging
import uuid
from typing import Optional

from src.cloud_controller.resource_manager import get_healthy_servers, update_server_state
from src.fault_tolerance.energy_scheduler import schedule_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tasks: dict = {}
_round_robin_counter = itertools.cycle([])
_server_list_cache: list = []


def _get_servers():
    return get_healthy_servers()


def assign_task_round_robin(task: dict) -> str:
    """Baseline: assign tasks to servers in round-robin order."""
    servers = _get_servers()
    if not servers:
        raise RuntimeError("No healthy servers available.")
    global _round_robin_counter, _server_list_cache
    if [s["server_id"] for s in servers] != [s["server_id"] for s in _server_list_cache]:
        _server_list_cache = servers
        _round_robin_counter = itertools.cycle(servers)
    server = next(_round_robin_counter)
    return server["server_id"]


def assign_task_ai(task: dict) -> str:
    """AI-enhanced: use energy-aware scheduler."""
    servers = _get_servers()
    return schedule_task(task, servers)


def submit_task(
    task_id: Optional[str] = None,
    priority: str = "normal",
    payload: Optional[dict] = None,
    use_ai: bool = True,
) -> dict:
    if task_id is None:
        task_id = str(uuid.uuid4())
    task = {"task_id": task_id, "priority": priority, "payload": payload or {}}
    if use_ai:
        server_id = assign_task_ai(task)
    else:
        server_id = assign_task_round_robin(task)

    _tasks[task_id] = {"task": task, "server_id": server_id, "status": "running"}
    logger.info("Task %s assigned to %s", task_id, server_id)
    return {"task_id": task_id, "server_id": server_id, "status": "running"}


def get_all_tasks() -> dict:
    return dict(_tasks)


def main():
    result = submit_task(priority="critical")
    logger.info("Submitted: %s", result)


if __name__ == "__main__":
    main()
