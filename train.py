import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO12s on the provided dataset.")
    parser.add_argument(
        "--weights", default="YOLO12s.pt", help="Path to pretrained YOLO12s weights."
    )
    parser.add_argument("--data", default="training.yaml", help="Dataset config file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. 0, 0,1 or cpu). Uses auto-selection when not set.",
    )
    parser.add_argument(
        "--project", default="runs/train", help="Project directory for checkpoints."
    )
    parser.add_argument("--name", default="yolo12s", help="Name of the training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    data_path = Path(args.data)

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
