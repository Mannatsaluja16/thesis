import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from src.fault_tolerance.vm_migration import (
    select_migration_target, migrate_vm, is_migration_needed,
)

_SERVERS = [
    {"server_id": "server_01", "cpu": 80.0, "mem": 70.0, "status": "healthy"},
    {"server_id": "server_02", "cpu": 30.0, "mem": 40.0, "status": "healthy"},
    {"server_id": "server_03", "cpu": 20.0, "mem": 25.0, "status": "healthy"},
]


def test_select_excludes_source():
    target = select_migration_target(_SERVERS, exclude="server_01")
    assert target != "server_01"


def test_select_returns_healthiest():
    target = select_migration_target(_SERVERS, exclude="server_01")
    # server_03 has lowest cpu+mem
    assert target == "server_03"


def test_select_raises_when_no_candidates():
    servers = [{"server_id": "s1", "cpu": 50, "mem": 50, "status": "healthy"}]
    with pytest.raises(ValueError):
        select_migration_target(servers, exclude="s1")


def test_migrate_vm_success():
    result = migrate_vm("server_01", "server_03", ["t1", "t2"])
    assert result.success is True
    assert result.source_id == "server_01"
    assert result.target_id == "server_03"
    assert result.workloads == ["t1", "t2"]


def test_is_migration_needed_true():
    pred = {"fault_predicted": True, "confidence": 0.8, "instance_id": "s1"}
    assert is_migration_needed(pred) is True


def test_is_migration_needed_false():
    pred = {"fault_predicted": False, "confidence": 0.3, "instance_id": "s1"}
    assert is_migration_needed(pred) is False
