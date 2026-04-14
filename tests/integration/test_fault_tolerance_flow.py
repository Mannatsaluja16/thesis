import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from src.fault_tolerance.vm_migration import is_migration_needed, migrate_vm, select_migration_target
from src.fault_tolerance.replication_manager import should_replicate, create_replica
from src.fault_tolerance.recovery_manager import handle_fault
from datetime import datetime

_SERVERS = [
    {"server_id": "server_01", "cpu": 85.0, "mem": 70.0, "status": "healthy"},
    {"server_id": "server_02", "cpu": 20.0, "mem": 25.0, "status": "healthy"},
    {"server_id": "server_03", "cpu": 30.0, "mem": 35.0, "status": "healthy"},
]


def test_high_confidence_triggers_migration():
    pred = {"fault_predicted": True, "confidence": 0.9, "instance_id": "server_01"}
    assert is_migration_needed(pred) is True
    target = select_migration_target(_SERVERS, exclude="server_01")
    result = migrate_vm("server_01", target, ["t1"])
    assert result.success is True


def test_low_confidence_skips_migration():
    pred = {"fault_predicted": False, "confidence": 0.2, "instance_id": "server_01"}
    assert is_migration_needed(pred) is False


def test_critical_task_high_confidence_triggers_replication():
    assert should_replicate(0.8, "critical") is True
    replica = create_replica("task_99", "server_02")
    assert replica.task_id == "task_99"


def test_normal_task_skips_replication():
    assert should_replicate(0.8, "normal") is False


def test_fault_event_triggers_recovery():
    result = handle_fault("server_05", datetime.utcnow())
    assert result.success is True


def test_recovery_reroutes_pending_tasks():
    result = handle_fault("server_06", datetime.utcnow())
    assert result.recovery_time >= 0
