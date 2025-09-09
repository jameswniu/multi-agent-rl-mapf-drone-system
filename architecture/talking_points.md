# Interview Talking Points

- **Core Idea**: A multi-agent reinforcement learning and path-finding system for drones that scores every action instantly on safety, efficiency, and smoothness.  
- **Architecture**: Three main agents (Ingestion, Preprocess, Prediction) supported by a Safety Controller and a Supervisor.  
- **Production Features**: Exposes FastAPI endpoints (/predict, /metrics, /healthz) for inference, monitoring, and health checks.  
- **Deployment Ready**: Can be trained locally, served via Docker, and deployed to Kubernetes with monitoring through Prometheus and Grafana.  
- **Value**: Combines RL research with real-world engineering practices to make AI safe, observable, and operational in real time.  
