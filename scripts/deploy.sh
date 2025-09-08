#!/usr/bin/env bash
# Deploys the app to Kubernetes
set -e
kubectl apply -f docker/k8s/deployment.yaml
kubectl apply -f docker/k8s/service.yaml
