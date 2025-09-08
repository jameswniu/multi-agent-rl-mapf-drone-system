#!/usr/bin/env bash
# Runs the FastAPI app locally using uvicorn
set -e
export $(grep -v '^#' .env | xargs)
uvicorn src.api.app:app --host 0.0.0.0 --port $API_PORT
