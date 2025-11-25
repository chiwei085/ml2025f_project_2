#!/usr/bin/env bash

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

$RUNNER infer.py --weights runs/train/project_2_res/weights/best.pt --name project_2_res
