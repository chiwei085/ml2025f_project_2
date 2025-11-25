#!/usr/bin/env bash

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

$RUNNER train.py --weights YOLO12s.pt --data training.yaml --epochs 250 --imgsz 640 --batch 16 --device 0 --name project_2_res
