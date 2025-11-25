import argparse
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO12s inference on test images.")
    parser.add_argument(
        "--weights",
        default="runs/train/project_2_res/weights/best.pt",
        help="Trained checkpoint path.",
    )
    parser.add_argument(
        "--source",
        default="datasets/test/images",
        help="Source images or directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="predict_txt",
        help="Directory to store prediction text files.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename for predictions (defaults to run name from weights).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device to run on (e.g. 0 or cpu).")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=1, help="Maximum detections per image.")
    return parser.parse_args()


def default_output_name(weights_path: Path, provided: str | None) -> str:
    if provided:
        return f"{Path(provided).stem}.txt"

    # Use the training run directory name when available, fallback to weight stem.
    try:
        run_name = weights_path.parents[1].name
    except IndexError:
        run_name = weights_path.stem

    return f"{run_name}.txt"


def infer(
    weights: Path,
    source: Path,
    output_dir: Path,
    output_name: str,
    imgsz: int,
    device: str | None,
    conf: float | None,
    iou: float | None,
    max_det: int,
) -> None:
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        save=True,
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
    )
    write_results(results, output_dir=output_dir, filename=output_name)


def write_results(
    results: Iterable[Results],
    output_dir: Path,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as f:
        for result in tqdm(results, desc="Inferencing", unit="img"):
            boxes = result.boxes
            if boxes is None or boxes.cls is None or boxes.cls.numel() == 0:
                continue

            file_stem = Path(result.path).stem
            labels = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            coords = boxes.xyxy.tolist()

            for label, conf_score, (x1, y1, x2, y2) in zip(labels, confs, coords, strict=False):
                line = (
                    f"{file_stem} {label} {conf_score:.4f} "
                    f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                )
                f.write(line)


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    output_name = default_output_name(weights_path, args.name)

    infer(
        weights=weights_path,
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        output_name=output_name,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
