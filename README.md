# Multi-Agent RL MAPF Drone System 🚀

Production-ready multi-agent reinforcement learning (MARL) system for drone pathfinding.
Includes containerized training, inference API, monitoring, and CI/CD.

## Features
- PPO-based MARL agent with modular design
- FastAPI inference service
- Dockerized + Kubernetes-ready
- Prometheus + Grafana monitoring
- GitHub Actions CI/CD

## Setup
```bash
pip install -r requirements.txt
bash scripts/run_training.sh
bash scripts/run_api.sh
```

API runs at: http://localhost:8000/predict
