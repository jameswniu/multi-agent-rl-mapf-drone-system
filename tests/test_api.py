import sys
import types
import pytest
from fastapi.testclient import TestClient

# ==================================================================
# ENVIRONMENT STUB
# ------------------------------------------------------------------
# The production app expects `src.env.drone_env.DroneEnv`.
# Importing the real env would trigger heavy dependencies (gym, etc).
# Instead, we stub it here with a dummy implementation.
# ==================================================================
class DummyEnv:
    def __init__(self, *_):
        self.observation_space = types.SimpleNamespace(shape=(2,))
        self.action_space = types.SimpleNamespace(n=2)

    def close(self):
        pass

# Register stubbed modules in sys.modules BEFORE importing app
env_pkg = types.ModuleType("src.env")
env_pkg.__path__ = []
env_mod = types.ModuleType("src.env.drone_env")
env_mod.DroneEnv = DummyEnv
sys.modules["src.env"] = env_pkg
sys.modules["src.env.drone_env"] = env_mod

# ==================================================================
# AGENT STUB
# ------------------------------------------------------------------
# The production app imports PPOAgent from `src.agents.ppo_agent`.
# That module transitively loads PyTorch. To avoid this in tests,
# we inject a lightweight stubbed agent with the same interface.
# ==================================================================
ppo_agent_mod = types.ModuleType("src.agents.ppo_agent")

class DummyAgent:
    def __init__(self, env):
        pass

    def load(self, path):  # model loading is skipped in tests
        pass

    def predict(self, state):  # always return a stub action
        return "stub_action"

ppo_agent_mod.PPOAgent = DummyAgent
sys.modules["src.agents.ppo_agent"] = ppo_agent_mod

# ==================================================================
# FIXTURE
# ------------------------------------------------------------------
# Provides a FastAPI TestClient with all stubs in place.
# Note: we import `app` here, AFTER stubbing dependencies.
# ==================================================================
@pytest.fixture
def client():
    from src.api.app import app
    with TestClient(app) as c:
        yield c

# ==================================================================
# TESTS
# ------------------------------------------------------------------
# Confirms that the /predict endpoint works with stubbed agent/env.
# Ensures isolation: no PyTorch, no heavy gym deps required.
# ==================================================================
def test_predict(client):
    payload = {"state": {"x": 1, "y": 2}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"action": "stub_action"}
