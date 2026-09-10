import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default="yolo26n.pt")
parser.add_argument("-r", "--resume", action="store_true")

args = parser.parse_args()

model = YOLO(args.model)
task = model.task if model.task is not None else "detect"

results = model.train(
    data="data.yaml",
    cfg=PROJECT_ROOT / "hyp.yaml",
    project=PROJECT_ROOT / "runs" / task,
    resume=args.resume,
)
