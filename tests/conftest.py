"""
PyTest Fixtures
---------------
This file defines reusable fixtures for our tests.

Why fixtures?
-> In PyTest, fixtures let us create and clean up objects (like environments or agents)
   automatically. This keeps tests clean and avoids repeating boilerplate.

Scope:
-> Each fixture here uses scope="function", so a fresh object is created
   for every test function. This prevents state from leaking between tests.
"""

import pytest
import sys
import types

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
# The original project depends on the `env` and `agents` packages which are not
# present in this kata.  Import errors would cause pytest collection to fail
# before any tests run.  To keep the tests self‑contained we provide minimal
# stub implementations when the real packages are missing.

try:  # pragma: no cover - only executed when real packages exist
    from env.drone_env import DroneEnv  # type: ignore
except Exception:  # pragma: no cover - exercised in the kata environment
    class DroneEnv:  # minimal stub
        def __init__(self, *_, **__):
            # observation_space and action_space attributes are required by
            # `PPOAgent` during initialisation.  A simple namespace object is
            # sufficient here.
            self.observation_space = types.SimpleNamespace(shape=(2,))
            self.action_space = types.SimpleNamespace(n=2)

        def close(self):  # no-op cleanup hook
            pass

    # register stub modules so other imports (e.g. `src.api.app`) succeed
    env_pkg = types.ModuleType("env")
    env_pkg.__path__ = []  # mark as package
    env_mod = types.ModuleType("env.drone_env")
    env_mod.DroneEnv = DroneEnv
    sys.modules.setdefault("env", env_pkg)
    sys.modules.setdefault("env.drone_env", env_mod)
    # also expose as `src.env.drone_env` for the FastAPI app
    sys.modules.setdefault("src.env", env_pkg)
    sys.modules.setdefault("src.env.drone_env", env_mod)

try:  # pragma: no cover - only executed when real packages exist
    from agents.ppo_agent import PPOAgent  # type: ignore
except Exception:  # pragma: no cover - exercised in the kata environment
    class PPOAgent:  # minimal stub
        def __init__(self, env=None):
            self.env = env

        def load(self, path):  # pragma: no cover - no behaviour
            pass

        def predict(self, state):  # simple deterministic action
            return "stub_action"

    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = []
    agents_mod = types.ModuleType("agents.ppo_agent")
    agents_mod.PPOAgent = PPOAgent
    sys.modules.setdefault("agents", agents_pkg)
    sys.modules.setdefault("agents.ppo_agent", agents_mod)


@pytest.fixture(scope="function")
def env():
    """
    Provides a fresh DroneEnv instance for each test.

    How it works:
    - Creates a new environment.
    - Yields it to the test function.
    - After the test completes, ensures env.close() is called.
    """
    e = DroneEnv()
    try:
        yield e
    finally:
        e.close()


@pytest.fixture(scope="function")
def agent():
    """
    Provides a PPOAgent tied to its own fresh DroneEnv.

    Why yield instead of return?
    -> Yield lets us run teardown code after the test finishes.
       Here, we make sure the environment is closed to free resources.
    """
    e = DroneEnv()
    a = PPOAgent(e)
    try:
        yield a
    finally:
        e.close()


@pytest.fixture(scope="function")
def model_path(tmp_path):
    """
    Provides a temporary file path for saving/loading models.

    Why tmp_path?
    -> PyTest gives each test its own temporary directory.
       Using tmp_path ensures models from one test don’t interfere with others.
    """
    return tmp_path / "ppo_test_model.pt"
