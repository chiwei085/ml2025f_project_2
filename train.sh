#!/usr/bin/env bash

DATA_PATH=${DATA_PATH:-training.yaml}
CONFIG_PATH=${CONFIG_PATH:-train_conf.yaml}

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

$RUNNER train.py --data "$DATA_PATH" --config "$CONFIG_PATH"
