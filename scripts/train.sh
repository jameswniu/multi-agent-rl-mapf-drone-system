#!/usr/bin/env bash
# Runs training with the specified config
set -e
python src/main.py --config configs/train.yaml
