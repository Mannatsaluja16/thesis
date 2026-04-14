import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

@pytest.fixture(scope="session")
def flask_client():
    from src.cloud_controller.api_gateway import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
