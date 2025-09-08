# Prometheus metrics definitions
# Import and use in app to record counts, latency, training stats

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["method", "endpoint"])
TRAINING_REWARD = Histogram("training_reward", "Reward per training episode")
