import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from src.fault_tolerance.replication_manager import should_replicate, create_replica, remove_replica


def test_should_replicate_high_confidence_critical():
    assert should_replicate(0.8, "critical") is True


def test_should_replicate_high_confidence_normal():
    assert should_replicate(0.8, "normal") is False


def test_should_replicate_low_confidence_critical():
    assert should_replicate(0.5, "critical") is False


def test_should_replicate_boundary():
    # Exactly at threshold is NOT > threshold
    assert should_replicate(0.75, "critical") is False


def test_create_and_remove_replica():
    replica = create_replica("task_01", "server_02")
    assert replica.task_id == "task_01"
    assert replica.target_server == "server_02"
    assert replica.active is True

    remove_replica(replica.replica_id)
    # No error raised


def test_remove_nonexistent_replica(caplog):
    remove_replica("nonexistent-id-9999")
    # Should log a warning, not raise
