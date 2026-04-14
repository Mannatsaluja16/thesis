import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from src.fault_tolerance.energy_scheduler import compute_energy_score, schedule_task


_SERVERS = [
    {"server_id": "s1", "cpu": 70.0, "mem": 60.0, "status": "healthy"},
    {"server_id": "s2", "cpu": 10.0, "mem": 12.0, "status": "healthy"},
    {"server_id": "s3", "cpu": 45.0, "mem": 50.0, "status": "healthy"},
]


def test_compute_energy_score_returns_float():
    score = compute_energy_score(_SERVERS[0])
    assert isinstance(score, float)


def test_compute_energy_score_range():
    for s in _SERVERS:
        score = compute_energy_score(s)
        assert 0.0 <= score <= 1.0


def test_schedule_task_returns_valid_server():
    task = {"task_id": "t1", "priority": "normal"}
    chosen = schedule_task(task, _SERVERS)
    assert chosen in [s["server_id"] for s in _SERVERS]


def test_schedule_task_prefers_low_load():
    task = {"task_id": "t2", "priority": "normal"}
    chosen = schedule_task(task, _SERVERS)
    # s2 has lowest load
    assert chosen == "s2"
