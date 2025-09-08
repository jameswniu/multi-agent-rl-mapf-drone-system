"""
Training Integration Tests
--------------------------
These tests check that training and inference run end-to-end
without crashing. They don’t test convergence (because RL is stochastic),
but they confirm that:
- The environment and agent connect properly.
- Training loop runs for a few episodes.
- The model can be saved and loaded.
- Inference (prediction) works on a trained agent.
"""

from main import train_and_save, run_inference


def test_train_and_inference(tmp_path):
    """
    Run a small training + inference cycle.

    Steps:
    1. Train agent for 2 episodes (short run, just for testing).
    2. Save model weights into a temporary path.
    3. Run inference for a few steps to confirm predictions.
    4. Assert that the model file exists.
    """
    # Temporary path for model
    model_path = tmp_path / "ppo_test.pt"

    # Step 1 -> Train agent
    agent, env = train_and_save(model_path, num_episodes=2)

    # Step 2 -> Run inference
    run_inference(agent, env, rollout_len=3)

    # Step 3 -> Check model file was saved
    assert model_path.exists()
