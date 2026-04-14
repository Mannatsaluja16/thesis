import logging
import random
import numpy as np
from src.config import RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_workload_trace(
    n_tasks: int = 500,
    n_steps: int = 1000,
) -> list:
    """
    Generate a list of (arrival_step, task_id, priority, cpu_demand, mem_demand).
    arrival_step is uniformly distributed over [0, n_steps).
    """
    tasks = []
    for i in range(n_tasks):
        arrival = random.randint(0, n_steps - 1)
        priority = random.choices(["normal", "critical"], weights=[0.8, 0.2])[0]
        cpu_demand = np.clip(np.random.normal(20, 10), 1, 80)
        mem_demand = np.clip(np.random.normal(15, 8),  1, 60)
        tasks.append({
            "task_id":    f"task_{i:04d}",
            "arrival":    arrival,
            "priority":   priority,
            "cpu_demand": round(cpu_demand, 1),
            "mem_demand": round(mem_demand, 1),
            "duration":   random.randint(5, 30),  # steps
        })
    tasks.sort(key=lambda t: t["arrival"])
    return tasks


def main():
    trace = generate_workload_trace()
    logger.info("Generated %d tasks", len(trace))
    logger.info("Sample tasks: %s", trace[:3])


if __name__ == "__main__":
    main()
