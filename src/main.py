"""
Main Training Script
--------------------
This file is the "orchestrator":
- It creates the environment (DroneEnv).
- It instantiates the PPO agent.
- It runs training for a few episodes.
- It saves the model to disk.
- It runs a quick inference demo.

Why separate this?
-> By keeping main.py clean and explicit, the workflow is easy to follow.
-> Environment logic stays in drone_env.py, agent logic stays in ppo_agent.py.
-> This separation mirrors good software engineering practice.
"""

import os
from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent
from integrity_stats import IntegrityStats  # tracks drift vs hallucination stats


def train_and_save(model_path="models/ppo_drone.pt", num_episodes=10):
    """
    End-to-end training routine.

    Steps:
    1. Create environment.
    2. Create PPO agent.
    3. Train agent for a given number of episodes.
    4. Save the trained model to disk.
    5. Print an integrity report.
    """
    stats = IntegrityStats()

    # Step 1 -> Environment
    env = DroneEnv()

    # Step 2 -> Agent
    agent = PPOAgent(env)

    # Monkey-patch validator to record stats after each check
    original_validate = agent.validator.validate
    def wrapped_validate(probs, value, action=None):
        errors = original_validate(probs, value, action)
        stats.record_policy(errors)
        return errors
    agent.validator.validate = wrapped_validate

    # Step 3 -> Train
    print(f"Starting training for {num_episodes} episodes...")
    agent.train(num_episodes=num_episodes)

    # Step 4 -> Save
    os.makedirs("models", exist_ok=True)
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Step 5 -> Report integrity stats
    stats.report(prefix="[Training Integrity Report]")

    return agent, env


def run_inference(agent, env, rollout_len=5):
    """
    Run a quick inference demo.
    - Reset the environment.
    - Let the trained agent pick greedy actions.
    - Print out actions and rewards.
    """
    stats = IntegrityStats()

    state, _ = env.reset()
    total_reward = 0.0

    for t in range(rollout_len):
        # Agent predicts best action (greedy)
        action = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)

        # Record env-level integrity stats
        stats.record_env(info)
        total_reward += reward

        print(f"Step {t+1}: action={env.action_map[action]}, reward={reward:.2f}")

        if terminated or truncated:
            break

    print(f"Total reward over {t+1} steps = {total_reward:.2f}")
    stats.report(prefix="[Inference Integrity Report]")


if __name__ == "__main__":
    # Run training and save model
    agent, env = train_and_save(num_episodes=10)

    # Run quick inference test
    run_inference(agent, env)
