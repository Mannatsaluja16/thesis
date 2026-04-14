import json
import random
from locust import HttpUser, task, between


class FaultToleranceUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def submit_task(self):
        priority = random.choice(["normal", "critical"])
        self.client.post(
            "/task/submit",
            data=json.dumps({"priority": priority}),
            headers={"Content-Type": "application/json"},
        )

    @task(2)
    def get_server_status(self):
        self.client.get("/servers/status")

    @task(1)
    def get_predictions(self):
        self.client.get("/predictions/latest")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics/summary")
