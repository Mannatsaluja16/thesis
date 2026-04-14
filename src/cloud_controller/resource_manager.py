import logging
import threading
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_lock = threading.Lock()

_servers: Dict[str, dict] = {
    f"server_{i:02d}": {
        "server_id":  f"server_{i:02d}",
        "cpu":        20.0 + i * 3,
        "mem":        30.0 + i * 2,
        "status":     "healthy",
        "vms":        [],
    }
    for i in range(1, 11)
}


def get_healthy_servers() -> List[dict]:
    with _lock:
        return [s for s in _servers.values() if s["status"] == "healthy"]


def get_all_servers() -> Dict[str, dict]:
    with _lock:
        return dict(_servers)


def update_server_state(server_id: str, updates: dict) -> None:
    with _lock:
        if server_id in _servers:
            _servers[server_id].update(updates)
            logger.debug("Updated server %s: %s", server_id, updates)
        else:
            logger.warning("Server %s not found.", server_id)


def get_server_load(server_id: str) -> Optional[dict]:
    with _lock:
        return _servers.get(server_id)


def add_vm_to_server(server_id: str, vm_id: str) -> None:
    with _lock:
        if server_id in _servers:
            if vm_id not in _servers[server_id]["vms"]:
                _servers[server_id]["vms"].append(vm_id)


def register_server(server_id: str, cpu: float = 0.0, mem: float = 0.0) -> None:
    with _lock:
        _servers[server_id] = {
            "server_id": server_id,
            "cpu": cpu,
            "mem": mem,
            "status": "healthy",
            "vms": [],
        }
        logger.info("Registered new server %s", server_id)


def main():
    logger.info("Healthy servers: %d", len(get_healthy_servers()))
    update_server_state("server_01", {"cpu": 85.0, "status": "at-risk"})
    logger.info("server_01 state: %s", get_server_load("server_01"))


if __name__ == "__main__":
    main()
