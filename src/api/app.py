from fastapi import FastAPI
from src.env.drone_env import DroneEnv
from src.agents.ppo_agent import PPOAgent

app = FastAPI(title="Drone Pathfinding API")

env = DroneEnv("configs/env.yaml")
agent = PPOAgent(env)
agent.load("models/ppo_drone.pt")

@app.post("/predict")
def predict(state: dict):
    action = agent.predict(state)
    return {"action": action}
