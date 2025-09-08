import sys
import types
import pytest
from fastapi.testclient import TestClient

# ===============================================================
# WHY DO WE NEED STUBS?
# ---------------------------------------------------------------
# In production, our FastAPI app imports:
#   - src.env.drone_env.DroneEnv (a Gym environment, heavy dep)
#   - src.agents.ppo_agent.PPOAgent (a PyTorch model, heavy dep)
#
# Problem: when we run API tests in CI, those libraries (gym,
# PyTorch) may not be installed. Importing them directly would
# crash the test collection with ModuleNotFoundError.
#
# Solution: we "stub" those modules with lightweight fake
# replacements. They have the same *interface* (same methods,
# same names) but skip all heavy logic. This way:
#   - The FastAPI app can import successfully.
#   - The /predict endpoint still works for tests.
#   - No GPU, model weights, or external deps are required.
# ===============================================================

# --- ENVIRONMENT STUB ---
# Fake DroneEnv just defines observation/action spaces.
class DummyEnv:
    def __init__(self, *_):
        self.observation_space = types.SimpleNamespace(shape=(2,))
        self.action_space = types.SimpleNamespace(n=2)

    def close(self):
        pass  # real env would clean up resources

# Register a fake "src.env" and "src.env.drone_env"
# so that when FastAPI imports them, it gets our dummy version.
env_pkg = types.ModuleType("src.env")
env_pkg.__path__ = []  # mark as package
env_mod = types.ModuleType("src.env.drone_env")
env_mod.DroneEnv = DummyEnv
sys.modules["src.env"] = env_pkg
sys.modules["src.env.drone_env"] = env_mod

# --- AGENT STUB ---
# Fake PPOAgent that pretends to load and predict.
ppo_agent_mod = types.ModuleType("src.agents.ppo_agent")

class DummyAgent:
    def __init__(self, env):
        pass  # ignore env
    def load(self, path):
        pass  # skip model loading
    def predict(self, state):
        return "stub_action"  # always same action for tests

ppo_agent_mod.PPOAgent = DummyAgent
sys.modules["src.agents.ppo_agent"] = ppo_agent_mod

# ===============================================================
# WHY IMPORT APP INSIDE FIXTURE?
# ---------------------------------------------------------------
# If we import src.api.app at the top of this file,
# it will try to pull in env + PPOAgent immediately.
# That would fail before our stubs are installed.
#
# Instead, we import the app only *after* stubbing,
# inside the pytest fixture. This ensures the app
# sees DummyEnv + DummyAgent, not the real heavy ones.
# ===============================================================

@pytest.fixture
def client():
    from src.api.app import app  # safe now, stubs are ready
    with TestClient(app) as c:   # context manager ensures
        yield c                  # startup/shutdown events fire

# ===============================================================
# TESTS
# ---------------------------------------------------------------
# What are we actually testing?
# - That the FastAPI /predict endpoint runs end-to-end
# - That it calls our DummyAgent.predict()
# - That it returns the expected JSON {"action": "stub_action"}
# ===============================================================

def test_predict(client):
    payload = {"state": {"x": 1, "y": 2}}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"action": "stub_action"}
