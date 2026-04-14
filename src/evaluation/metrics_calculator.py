import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IDLE_POWER   = 100.0
ACTIVE_COEFF = 150.0
EC2_RATE_PER_HR = 0.023


class MetricsCollector:
    def __init__(self, n_servers: int = 10):
        self.n_servers    = n_servers
        self.step_records: List[dict] = []

    def record_step(self, step: int, servers: list, completed: int, failed: int):
        avg_cpu = sum(s["cpu_used"] for s in servers) / max(self.n_servers, 1)
        self.step_records.append({
            "step":      step,
            "completed": completed,
            "failed":    failed,
            "avg_cpu":   avg_cpu,
        })


def compute_energy(servers: list, duration_steps: int) -> float:
    """Total energy in Wh across all servers."""
    total_wh = 0.0
    for srv in servers:
        load  = srv.get("cpu_used", 0) / (8 * 100)
        power = IDLE_POWER + load * ACTIVE_COEFF
        total_wh += power * (duration_steps / 3600)
    return round(total_wh, 2)


def compute_reliability(total_tasks: int, failed_tasks: int) -> float:
    if total_tasks == 0:
        return 100.0
    return round((total_tasks - failed_tasks) / total_tasks * 100, 2)


def compute_throughput(completed_tasks: int, duration_steps: int) -> float:
    return round(completed_tasks / max(duration_steps, 1), 4)


def compute_cost(n_servers: int, duration_steps: int) -> float:
    return round(n_servers * (duration_steps / 3600) * EC2_RATE_PER_HR, 4)


def compute_all_metrics(
    servers: list,
    total_tasks: int,
    completed_tasks: int,
    failed_tasks: int,
    duration_steps: int,
    recovery_times: list,
    latency_ms_list: list,
) -> dict:
    return {
        "energy_wh":       compute_energy(servers, duration_steps),
        "execution_time_s": duration_steps,
        "latency_ms":      round(sum(latency_ms_list) / len(latency_ms_list), 2) if latency_ms_list else 0.0,
        "throughput":      compute_throughput(completed_tasks, duration_steps),
        "reliability_pct": compute_reliability(total_tasks, failed_tasks),
        "cost_usd":        compute_cost(len(servers), duration_steps),
        "recovery_time_s": round(sum(recovery_times) / len(recovery_times), 2) if recovery_times else 0.0,
    }
