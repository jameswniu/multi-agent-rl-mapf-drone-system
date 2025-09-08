# This Makefile defines shortcuts for common developer tasks
# Instead of remembering long commands, you type "make test" etc.

.PHONY: test build run install

install:
	pip install -e .

test:
	pytest --cov=src

build:
	docker build -t drone-system -f docker/Dockerfile.prod .

run:
	docker-compose up
