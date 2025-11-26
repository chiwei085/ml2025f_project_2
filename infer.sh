#!/usr/bin/env bash

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

# CONF=${CONF:-0.1}
# IOU=${IOU:-0.5}

$RUNNER infer.py \
  --weights runs/train/project_2_res2/weights/best.pt \
  --name project_2_res \
  # --conf "$CONF" \
  # --iou "$IOU"
