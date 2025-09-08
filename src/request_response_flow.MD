1\. Client sends a request



Example:

POST /predict with JSON body:



{ "state": { "drone": \[0, 0], "goal": \[5, 5] } }



2\. Middleware intercepts



The add\_metrics middleware runs before the request reaches /predict.



It records the start time.



After the endpoint finishes, it calculates how long the request took.



It increments two Prometheus metrics:



api\_requests\_total{method="POST", endpoint="/predict"}



api\_request\_latency\_seconds{method="POST", endpoint="/predict"}



It also writes a log entry like:



2025-09-08 12:34:56 - src.api.app - INFO - POST /predict completed in 0.032s



3\. The /predict endpoint runs



The request body is parsed into a Python dictionary.



The PPO agent (PPOAgent) calls its predict() method with this state.



If it succeeds, the action is returned as JSON:



{ "action": "move\_up" }





If something fails (e.g., invalid input), an APIError is raised.



The error handler converts it into a structured JSON error:



{ "error": "Prediction failed: invalid state" }





Status code 500 is sent back.



4\. Response leaves API



The middleware (from step 2) finishes by logging the total time and sending the response back to the client.



The metrics are already updated, so Prometheus will see them at the next scrape.



5\. Monitoring systems interact



Prometheus periodically calls GET /metrics.



It collects counters/histograms, which include how many requests succeeded, how long they took, etc.



Kubernetes (or another orchestrator) calls GET /healthz.



If it gets { "status": "ok" }, it considers the service alive.



If this fails repeatedly, Kubernetes restarts the container.

