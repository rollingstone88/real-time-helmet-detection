"""Generate a tiny *synthetic* helmet dataset (Pascal VOC) and a test video.

This is ONLY a pipeline sanity check (does training / evaluation / video inference run
end-to-end?). The cartoon images are not a substitute for real data; see README for the
real datasets.

    python tools/make_synthetic_dataset.py --out data/synthetic --num-images 80
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np


def draw_scene(rng: np.random.Generator, w: int = 320, h: int = 240, max_riders: int = 3):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rng.integers(90, 160, 3)  # sky / background
    cv2.rectangle(img, (0, int(h * 0.65)), (w, h), tuple(int(c) for c in rng.integers(40, 80, 3)), -1)  # road
    objects = []
    for _ in range(int(rng.integers(1, max_riders + 1))):
        cx = int(rng.integers(40, w - 40))
        base = int(rng.integers(int(h * 0.7), h - 10))
        s = float(rng.uniform(0.7, 1.3))
        # motorcycle
        cv2.circle(img, (cx - int(22 * s), base), int(10 * s), (20, 20, 20), -1)
        cv2.circle(img, (cx + int(22 * s), base), int(10 * s), (20, 20, 20), -1)
        cv2.rectangle(img, (cx - int(25 * s), base - int(18 * s)), (cx + int(25 * s), base - int(6 * s)),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        # rider body
        cv2.rectangle(img, (cx - int(8 * s), base - int(55 * s)), (cx + int(8 * s), base - int(15 * s)),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        # head
        r = int(9 * s)
        hc = (cx, base - int(55 * s) - r)
        helmet = bool(rng.random() < 0.5)
        cv2.circle(img, hc, r, (140, 180, 225), -1)  # skin (BGR)
        if helmet:
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.ellipse(img, hc, (r + 3, r + 3), 0, 180, 360, color, -1)  # shell
            cv2.rectangle(img, (hc[0] - r - 3, hc[1] - 1), (hc[0] + r + 3, hc[1] + 2), color, -1)
            box = [hc[0] - r - 3, hc[1] - r - 3, hc[0] + r + 3, hc[1] + r]
        else:
            cv2.ellipse(img, hc, (r, r), 0, 200, 340, (20, 20, 30), 3)  # hair
            box = [hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r]
        box = [int(np.clip(box[0], 0, w - 1)), int(np.clip(box[1], 0, h - 1)),
               int(np.clip(box[2], 0, w - 1)), int(np.clip(box[3], 0, h - 1))]
        objects.append(("With Helmet" if helmet else "Without Helmet", box))
    noise = rng.normal(0, 4, img.shape)
    return (img + noise).clip(0, 255).astype(np.uint8), objects


def voc_xml(filename: str, w: int, h: int, objects) -> str:
    objs = "".join(
        f"<object><name>{escape(n)}</name><pose>Unspecified</pose><truncated>0</truncated>"
        f"<occluded>0</occluded><difficult>0</difficult><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for n, b in objects)
    return (f"<annotation><folder>images</folder><filename>{filename}</filename><size><width>{w}</width>"
            f"<height>{h}</height><depth>3</depth></size>{objs}</annotation>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--num-images", type=int, default=80)
    ap.add_argument("--video-frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "annotations").mkdir(parents=True, exist_ok=True)
    for i in range(args.num_images):
        img, objects = draw_scene(rng)
        name = f"synthetic_{i:04d}.png"
        cv2.imwrite(str(args.out / "images" / name), img)
        (args.out / "annotations" / f"synthetic_{i:04d}.xml").write_text(
            voc_xml(name, img.shape[1], img.shape[0], objects))
    video = cv2.VideoWriter(str(args.out / "test_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(args.video_frames):
        video.write(draw_scene(rng)[0])
    video.release()
    print(f"Wrote {args.num_images} images + test_video.mp4 to {args.out}")


if __name__ == "__main__":
    main()
