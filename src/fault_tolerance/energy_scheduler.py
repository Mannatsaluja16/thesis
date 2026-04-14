import logging
import boto3
from src.config import (
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_INSTANCE_TYPE, AWS_AMI_ID, AWS_KEY_PAIR_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Power model constants
IDLE_POWER   = 100.0   # watts
ACTIVE_COEFF = 150.0   # watts at 100% load above idle
CPU_COEFF    = 0.6
MEM_COEFF    = 0.4
MIGRATION_OVERHEAD = 5.0   # watts per migration event
CONSOLIDATION_THRESHOLD = 0.15   # servers below 15% load are idle
SCALE_OUT_THRESHOLD     = 0.85   # servers above 85% load triggers scale-out


def compute_energy_score(server: dict) -> float:
    """
    Score = cpu_load * CPU_COEFF + mem_load * MEM_COEFF
    Lower score = more energy-efficient.
    """
    cpu = server.get("cpu", 0) / 100.0
    mem = server.get("mem", 0) / 100.0
    return cpu * CPU_COEFF + mem * MEM_COEFF


def schedule_task(task: dict, servers: list) -> str:
    """
    Assign task to the lowest-score healthy server with enough capacity.
    Triggers scale-out if all servers are near capacity.
    Returns server_id.
    """
    candidates = [
        s for s in servers
        if s["status"] == "healthy"
        and s["cpu"] < SCALE_OUT_THRESHOLD * 100
    ]
    if not candidates:
        logger.warning("All servers near capacity — triggering scale-out.")
        _scale_out()
        candidates = [s for s in servers if s["status"] == "healthy"]

    best = min(candidates, key=compute_energy_score)
    logger.info("Scheduled task %s → server %s", task.get("task_id"), best["server_id"])
    return best["server_id"]


def consolidate_idle_servers(servers: list) -> list:
    """Return list of server_ids that should be suspended (load < threshold)."""
    idle = [
        s["server_id"]
        for s in servers
        if (s.get("cpu", 100) / 100.0) < CONSOLIDATION_THRESHOLD
        and s["status"] == "healthy"
    ]
    if idle:
        logger.info("Consolidating idle servers: %s", idle)
    return idle


def _scale_out():
    """Launch a new EC2 t3.micro instance (real AWS call)."""
    try:
        ec2 = boto3.client(
            "ec2",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        resp = ec2.run_instances(
            ImageId=AWS_AMI_ID,
            InstanceType=AWS_INSTANCE_TYPE,
            KeyName=AWS_KEY_PAIR_NAME,
            MinCount=1,
            MaxCount=1,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "fault-tolerance-scale-out"}],
            }],
        )
        iid = resp["Instances"][0]["InstanceId"]
        logger.info("Scale-out: launched new instance %s", iid)
    except Exception as e:
        logger.error("Scale-out failed: %s", e)


def main():
    servers = [
        {"server_id": "s1", "cpu": 70.0, "mem": 60.0, "status": "healthy"},
        {"server_id": "s2", "cpu": 10.0, "mem": 12.0, "status": "healthy"},
        {"server_id": "s3", "cpu": 45.0, "mem": 50.0, "status": "healthy"},
    ]
    task = {"task_id": "t001", "priority": "normal"}
    chosen = schedule_task(task, servers)
    logger.info("Chosen server: %s", chosen)

    idle = consolidate_idle_servers(servers)
    logger.info("Idle servers to suspend: %s", idle)


if __name__ == "__main__":
    main()
