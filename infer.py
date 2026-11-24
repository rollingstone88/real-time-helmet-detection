"""Run helmet detection on recorded video, a live webcam / RTSP stream, or images.

    python infer.py --weights runs/exp/best.pt --source traffic.mp4 --show
    python infer.py --weights runs/exp/best.pt --source 0 --show                 # webcam
    python infer.py --weights runs/exp/best.pt --source rtsp://camera/stream --show
    python infer.py --weights runs/exp/best.pt --source photos/                  # image folder

Keys while the window is open: q / Esc = quit, p = pause.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import helmet  # noqa: F401
import cv2

from helmet.config import apply_overrides, load_config
from helmet.dataset import IMAGE_EXTENSIONS
from helmet.model import load_checkpoint
from helmet.pipeline import FrameResult, HelmetPipeline
from helmet.utils import select_device
from helmet.visualize import draw_results

log = logging.getLogger("infer")


def source_kind(source: str) -> str:
    if source.isdigit():
        return "webcam"
    if "://" in source:
        return "stream"
    p = Path(source)
    if p.is_dir():
        return "images"
    if p.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"
    if not p.exists():
        raise SystemExit(f"Source not found: {source}")
    return "video"


def csv_rows(frame_idx: int, t: float, result: FrameResult):
    for h in result.heads:
        yield [frame_idx, f"{t:.3f}", "head", "", h.name, f"{h.score:.4f}", *[f"{v:.1f}" for v in h.box]]
    for r in result.riders:
        yield [frame_idx, f"{t:.3f}", "rider", r.track_id or "", r.status, f"{r.score:.4f}",
               *[f"{v:.1f}" for v in r.box]]


def run_images(pipeline: HelmetPipeline, paths, out_dir: Path, show: bool, writer):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("Could not read %s", path)
            continue
        result = pipeline.process(frame)
        vis = draw_results(frame, result, show_riders=pipeline.rider_detector is not None)
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), vis)
        if writer:
            writer.writerows([[path.name, *row[1:]] for row in csv_rows(i, 0.0, result)])
        log.info("%s: %d heads, %d riders (%.0f ms)", path.name, len(result.heads), len(result.riders),
                 result.inference_ms)
        if show:
            cv2.imshow("Helmet detection", vis)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    log.info("Annotated images saved to %s", out_dir)


def run_video(pipeline: HelmetPipeline, source, kind: str, output: Path, show: bool, every: int,
              max_frames: int, writer_csv):
    cap = cv2.VideoCapture(int(source) if kind == "webcam" else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source!r}")
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_fps = in_fps if 1 <= in_fps <= 120 else 25.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_idx, fps, result, paused = 0, None, FrameResult(), False
    t_start = time.perf_counter()
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                loop_start = time.perf_counter()
                if frame_idx % every == 0:
                    result = pipeline.process(frame)
                    if writer_csv:
                        writer_csv.writerows(csv_rows(frame_idx, frame_idx / in_fps, result))
                vis = draw_results(frame, result, fps, pipeline.violation_count,
                                   show_riders=pipeline.rider_detector is not None)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (w, h))
                writer.write(vis)
                inst = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst  # smoothed processing FPS
                frame_idx += 1
                if frame_idx % 100 == 0:
                    log.info("frame %d | %.1f FPS", frame_idx, fps)
                if max_frames and frame_idx >= max_frames:
                    break
            if show:
                cv2.imshow("Helmet detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    log.info("Processed %d frames in %.1fs (%.1f FPS average)", frame_idx, elapsed, frame_idx / max(elapsed, 1e-6))
    if pipeline.violation_count is not None:
        log.info("Unique riders without a helmet: %d", pipeline.violation_count)
    log.info("Annotated video saved to %s", output)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, help="video file, webcam index (0), rtsp/http URL, image or folder")
    ap.add_argument("--config", help="config whose 'inference' section is used (default: training config)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--output", type=Path, help="output video / folder (default: runs/infer/...)")
    ap.add_argument("--show", action="store_true", help="display results in a window")
    ap.add_argument("--conf", type=float, help="confidence threshold")
    ap.add_argument("--no-riders", action="store_true", help="only detect helmet / no-helmet heads")
    ap.add_argument("--no-track", action="store_true", help="disable rider tracking")
    ap.add_argument("--enhance-low-light", action="store_true", help="CLAHE on dark frames")
    ap.add_argument("--every", type=int, default=1, help="run the detector every N frames (speed-up)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--csv", type=Path, help="write every detection to a CSV file")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config, args.set) if args.config else apply_overrides(load_checkpoint(args.weights)["config"], args.set)
    inf = cfg.get("inference", {})
    kind = source_kind(args.source)
    riders = dict(inf.get("riders", {}))
    tracking = dict(inf.get("tracking", {}))
    low_light = dict(inf.get("low_light", {}))
    if args.no_riders:
        riders["enabled"] = False
    if args.no_track or kind in ("image", "images"):
        tracking["enabled"] = False  # tracking is meaningless across unrelated images
    if args.enhance_low_light:
        low_light["enhance"] = True

    device = select_device(args.device)
    pipeline = HelmetPipeline(str(args.weights), device,
                              conf_threshold=args.conf if args.conf is not None else inf.get("conf_threshold", 0.5),
                              cross_class_nms_iou=inf.get("cross_class_nms_iou", 0.5), riders=riders,
                              tracking=tracking, low_light=low_light,
                              violation_class=inf.get("violation_class", "no_helmet"))
    log.info("Device %s | classes %s | riders %s | tracking %s", device, pipeline.class_names,
             pipeline.rider_detector is not None, pipeline.tracker is not None)

    csv_file = writer_csv = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(["frame", "time_s", "type", "track_id", "label", "score", "x1", "y1", "x2", "y2"])
    try:
        if kind in ("image", "images"):
            src = Path(args.source)
            paths = [src] if kind == "image" else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            run_images(pipeline, paths, args.output or Path("runs/infer") / (src.stem if kind == "image" else src.name),
                       args.show, writer_csv)
        else:
            name = Path(args.source).stem if kind == "video" else f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"
            run_video(pipeline, args.source, kind, args.output or Path("runs/infer") / f"{name}_pred.mp4",
                      args.show, max(1, args.every), args.max_frames, writer_csv)
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
