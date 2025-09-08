\# API Reference



\## POST /predict

Request body: `{ "state": { ... } }`  

Response: `{ "action": ... }`



\## GET /metrics

Prometheus-compatible metrics, scrapeable by monitoring systems.



\## GET /healthz

Simple health check endpoint, returns 200 if alive.



