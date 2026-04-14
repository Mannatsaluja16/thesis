import logging
import os
import time
from flask import Flask, jsonify, request, render_template

from src.cloud_controller.task_scheduler import submit_task, get_all_tasks
from src.cloud_controller.resource_manager import get_all_servers
from src.cloud_controller.monitoring_system import (
    start_monitoring, get_latest_predictions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tmpl_dir = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=_tmpl_dir)

# Start background monitoring when loaded by Gunicorn or directly.
# Skipped during pytest runs (PYTEST_CURRENT_TEST is set automatically by pytest).
if not os.environ.get("PYTEST_CURRENT_TEST"):
    start_monitoring(interval_seconds=30)

# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #

@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/task/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    if "task_id" not in data and "priority" not in data and not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    priority = data.get("priority", "normal")
    if priority not in ("normal", "critical"):
        return jsonify({"error": "priority must be 'normal' or 'critical'."}), 400

    result = submit_task(
        task_id=data.get("task_id"),
        priority=priority,
        payload=data.get("payload"),
        use_ai=True,
    )
    return jsonify(result), 201


@app.route("/servers/status", methods=["GET"])
def servers_status():
    servers = get_all_servers()
    return jsonify({"servers": list(servers.values())}), 200


@app.route("/predictions/latest", methods=["GET"])
def latest_predictions():
    return jsonify({"predictions": get_latest_predictions()}), 200


@app.route("/metrics/summary", methods=["GET"])
def metrics_summary():
    servers = list(get_all_servers().values())
    total   = len(servers)
    healthy = sum(1 for s in servers if s["status"] == "healthy")
    tasks   = get_all_tasks()
    total_tasks   = len(tasks)
    failed_tasks  = sum(1 for t in tasks.values() if t["status"] == "failed")
    reliability   = round((total_tasks - failed_tasks) / total_tasks * 100, 2) if total_tasks else 100.0

    avg_cpu = round(sum(s["cpu"] for s in servers) / total, 2) if total else 0.0
    avg_mem = round(sum(s.get("mem", 0) for s in servers) / total, 2) if total else 0.0

    energy_wh = sum(
        (100 + s["cpu"] * 1.5) / 3600  # simplified; 1 second tick
        for s in servers
    )

    return jsonify({
        "energy_consumption_wh": round(energy_wh, 4),
        "execution_time_s": None,
        "latency_ms": None,
        "throughput_tasks_per_s": None,
        "reliability_pct": reliability,
        "cost_usd": round(total * 0.023 / 3600, 6),
        "recovery_time_s": None,
        "total_servers": total,
        "healthy_servers": healthy,
        "avg_cpu_pct": avg_cpu,
        "avg_mem_pct": avg_mem,
    }), 200


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
