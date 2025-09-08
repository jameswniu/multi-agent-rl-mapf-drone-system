import sys
import types
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Provide a TestClient with the agent predict method stubbed.

    The production application initialises a learning environment and loads a
    trained model at import time.  For isolated unit tests we replace those
    heavy dependencies with lightweight stubs before the app is imported.
    """
    # ------------------------------------------------------------------
    # Stub out the environment module expected by the FastAPI app
    # ------------------------------------------------------------------
    class DummyEnv:
        def __init__(self, *_, **__):
            self.observation_space = types.SimpleNamespace(shape=(2,))
            self.action_space = types.SimpleNamespace(n=2)

        def close(self):
            pass

    env_pkg = types.ModuleType("src.env")
    env_pkg.__path__ = []
    env_mod = types.ModuleType("src.env.drone_env")
    env_mod.DroneEnv = DummyEnv
    sys.modules.setdefault("src.env", env_pkg)
    sys.modules.setdefault("src.env.drone_env", env_mod)

    # ------------------------------------------------------------------
    # Replace PPOAgent with a lightweight stub
    # ------------------------------------------------------------------
    import src.agents.ppo_agent as ppo_agent

    class DummyAgent:
        def __init__(self, env):
            pass

        def load(self, path):  # model loading is a no-op in tests
            pass

        def predict(self, state):
            return "stub_action"

    monkeypatch.setattr(ppo_agent, "PPOAgent", DummyAgent)

    # Import the app *after* stubbing dependencies
    from src.api.app import app

    with TestClient(app) as c:
        yield c


def test_predict(client):
    response = client.post("/predict", json={"state": {"x": 1, "y": 2}})
    assert response.status_code == 200
    assert response.json() == {"action": "stub_action"}
