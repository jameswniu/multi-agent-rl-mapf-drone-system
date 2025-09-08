# Makefile for multi-agent-rl-mapf-drone-system
# Provides shortcuts for common developer tasks

.PHONY: install test coverage lint format clean build run deploy

# Install project in editable mode
install:
	pip install -e .

# Run tests with pytest
test:
	pytest -v

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=term-missing

# Lint the code with flake8
lint:
	flake8 src tests

# Format code with black
format:
	black src tests

# Clean build/test artifacts
clean:
	rm -rf build dist .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

# Build Docker image
build:
	docker build -t drone-system -f docker/Dockerfile.prod .

# Run services with docker-compose
run:
	docker-compose -f docker/docker-compose.yml up

# Deploy using Kubernetes manifests
deploy:
	kubectl apply -f docker/k8s/
