import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from datetime import datetime, timedelta
from src.fault_tolerance.recovery_manager import (
    calculate_recovery_time, handle_fault, restore_from_replica,
)


def test_calculate_recovery_time_positive():
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=10)
    assert calculate_recovery_time(t0, t1) == pytest.approx(10.0, abs=0.5)


def test_calculate_recovery_time_zero_when_equal():
    t = datetime.utcnow()
    assert calculate_recovery_time(t, t) == pytest.approx(0.0, abs=0.1)


def test_handle_fault_sets_recovering_then_healthy():
    result = handle_fault("server_99", datetime.utcnow())
    assert result.success is True
    assert result.recovery_time >= 0.0


def test_restore_from_replica_returns_true():
    assert restore_from_replica("server_01") is True
