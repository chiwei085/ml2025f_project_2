#!/usr/bin/env bash

uv run train.py --weights YOLO12s.pt --data training.yaml --epochs 250 --imgsz 640 --batch 16 --name project_2_res

