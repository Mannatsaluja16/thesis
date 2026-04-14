import logging
import os
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.config import (
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS = [
    "CPUUtilization",
    "NetworkIn",
    "NetworkOut",
    "DiskReadOps",
    "DiskWriteOps",
    "StatusCheckFailed",
]


def _get_client():
    return boto3.client(
        "cloudwatch",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def collect_metrics(instance_id: str, start_time, end_time) -> pd.DataFrame:
    """Pull CloudWatch metrics for a single EC2 instance."""
    client = _get_client()
    rows = {}
    for metric_name in METRICS:
        try:
            response = client.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName=metric_name,
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=60,
                Statistics=["Average"],
            )
            for dp in response["Datapoints"]:
                ts = dp["Timestamp"].replace(tzinfo=timezone.utc)
                if ts not in rows:
                    rows[ts] = {"timestamp": ts, "instance_id": instance_id}
                rows[ts][metric_name] = dp["Average"]
        except Exception as e:
            logger.error("Error collecting %s for %s: %s", metric_name, instance_id, e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(list(rows.values())).sort_values("timestamp").reset_index(drop=True)
    for col in METRICS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def collect_all_instances(instance_ids: list, duration_minutes: int = 60) -> pd.DataFrame:
    """Collect metrics for multiple instances."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=duration_minutes)
    frames = []
    for iid in instance_ids:
        logger.info("Collecting metrics for %s", iid)
        df = collect_metrics(iid, start_time, end_time)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    os.makedirs("data/raw", exist_ok=True)
    path = os.path.join("data/raw", filename)
    df.to_csv(path, index=False)
    logger.info("Saved raw data to %s", path)


def main():
    ec2 = boto3.client(
        "ec2",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    resp = ec2.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )
    instance_ids = [
        i["InstanceId"]
        for r in resp["Reservations"]
        for i in r["Instances"]
    ]
    if not instance_ids:
        logger.warning("No running instances found. Use synthetic generator instead.")
        return
    df = collect_all_instances(instance_ids, duration_minutes=60)
    if not df.empty:
        save_raw_data(df, "cloudwatch_metrics.csv")
    else:
        logger.warning("No data collected.")


if __name__ == "__main__":
    main()
