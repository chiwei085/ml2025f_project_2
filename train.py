import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with a config-driven setup.")
    parser.add_argument("--data", default="training.yaml", help="Dataset config file.")
    parser.add_argument(
        "--config",
        default="train_conf.yaml",
        help="YAML file holding training parameters (weights, epochs, etc.).",
    )
    return parser.parse_args()


def load_train_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(config).__name__}")

    return config


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    config_path = Path(args.config)

    train_config = load_train_config(config_path)
    weights_path = Path(train_config.pop("weights", "yolo12s.pt"))

    train_config = {k: v for k, v in train_config.items() if v is not None}
    train_config["data"] = str(data_path)

    model = YOLO(str(weights_path))
    model.train(**train_config)


if __name__ == "__main__":
    main()
