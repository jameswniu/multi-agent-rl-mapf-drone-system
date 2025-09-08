from src.env.drone_env import DroneEnv
from src.agents.ppo_agent import PPOAgent

if __name__ == "__main__":
    env = DroneEnv(config_path="configs/env.yaml")
    agent = PPOAgent(env, config_path="configs/train.yaml")
    agent.train(num_episodes=1000)
    agent.save("models/ppo_drone.pt")
