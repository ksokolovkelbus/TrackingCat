from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Export TrackingCat YOLO model to OpenVINO for Intel CPU inference.")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 800])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for imgsz in args.sizes:
        dest = Path(f"yolo26s_{imgsz}_openvino_model")
        if dest.exists() and not args.force:
            print(f"Skip existing {dest} (use --force to overwrite)")
            continue
        shutil.rmtree(dest, ignore_errors=True)
        shutil.rmtree("yolo26s_openvino_model", ignore_errors=True)
        print(f"Exporting {args.model} imgsz={imgsz} -> {dest}", flush=True)
        out = YOLO(args.model).export(
            format="openvino",
            imgsz=imgsz,
            dynamic=False,
            half=False,
            int8=False,
            simplify=True,
        )
        shutil.move(str(out).rstrip("/"), dest)
        print(f"Saved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
