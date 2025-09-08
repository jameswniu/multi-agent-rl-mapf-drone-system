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
from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent


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
