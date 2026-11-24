"""Box-aware data augmentation (numpy / OpenCV).

Every transform takes and returns ``(image, target)`` where ``image`` is an RGB uint8
array (H, W, 3) and ``target`` holds absolute ``xyxy`` boxes plus per-box arrays.

The augmentations are chosen for the failure modes of roadside helmet detection:

* lighting   : brightness/contrast, gamma (night / over-exposure), HSV colour jitter
* occlusion  : random patches pasted over part of a head (riders behind riders, hands, poles)
* motion     : Gaussian / motion blur from fast riders and low shutter speeds
* scale      : random zoom-crops so far-away (small) and close (large) riders are seen
* sensor     : Gaussian noise (cheap CCTV / night footage)

The ``stress_*`` functions are deterministic versions used only at evaluation time to
measure robustness to the same conditions.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .dataset import filter_target


def _rng(rng: Optional[np.random.Generator]):
    return rng if rng is not None else np.random  # global stream is seeded per worker


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        return "Compose(" + ", ".join(type(t).__name__ for t in self.transforms) + ")"


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        w = image.shape[1]
        image = np.ascontiguousarray(image[:, ::-1])
        boxes = target["boxes"].copy()
        boxes[:, [0, 2]] = w - target["boxes"][:, [2, 0]]
        return image, {**target, "boxes": boxes}


class RandomScaleCrop:
    """Zoom in by cropping a random window covering ``scale`` of each side.

    A crop is only accepted if every box is either kept with at least ``min_visibility``
    of its area or falls completely outside, so we never keep a sliver of a head that is
    labelled, nor leave a mostly-visible head unlabelled.
    """

    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.6, 1.0),
                 min_visibility: float = 0.6, max_tries: int = 20):
        self.p, self.scale, self.min_visibility, self.max_tries = p, tuple(scale), min_visibility, max_tries

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        h, w = image.shape[:2]
        boxes = target["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for _ in range(self.max_tries):
            s = np.random.uniform(*self.scale)
            cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
            x0 = np.random.randint(0, w - cw + 1)
            y0 = np.random.randint(0, h - ch + 1)
            nb = boxes.copy()
            nb[:, [0, 2]] = (nb[:, [0, 2]] - x0).clip(0, cw)
            nb[:, [1, 3]] = (nb[:, [1, 3]] - y0).clip(0, ch)
            new_areas = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
            vis = np.divide(new_areas, areas, out=np.zeros_like(areas), where=areas > 0)
            keep = vis >= self.min_visibility
            if np.any((vis > 0) & ~keep):
                continue  # would cut a head partially -> try another window
            if len(boxes) and not keep.any():
                continue  # keep at least one object
            crop = np.ascontiguousarray(image[y0:y0 + ch, x0:x0 + cw])
            return crop, filter_target({**target, "boxes": nb}, keep)
        return image, target


class RandomBrightnessContrast:
    def __init__(self, p: float = 0.5, brightness: float = 0.25, contrast: float = 0.25):
        self.p, self.brightness, self.contrast = p, brightness, contrast

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * np.random.uniform(-self.brightness, self.brightness)
        out = image.astype(np.float32) * alpha + beta
        return out.clip(0, 255).astype(np.uint8), target


class RandomGamma:
    """gamma > 1 darkens (dusk / night), gamma < 1 brightens (glare / over-exposure)."""

    def __init__(self, p: float = 0.3, gamma: Tuple[float, float] = (0.5, 2.2)):
        self.p, self.gamma = p, tuple(gamma)

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        # sample in log-space so darkening and brightening are equally likely
        g = float(np.exp(np.random.uniform(np.log(self.gamma[0]), np.log(self.gamma[1]))))
        return apply_gamma(image, g), target


class RandomHSV:
    def __init__(self, p: float = 0.5, hue: float = 0.015, saturation: float = 0.5, value: float = 0.3):
        self.p, self.hue, self.saturation, self.value = p, hue, saturation, value

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-self.hue, self.hue) * 180.0) % 180.0
        hsv[..., 1] *= 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hsv[..., 2] *= 1.0 + np.random.uniform(-self.value, self.value)
        hsv[..., 1:] = hsv[..., 1:].clip(0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), target


class RandomBlur:
    def __init__(self, p: float = 0.2, max_kernel: int = 9):
        self.p, self.max_kernel = p, max(3, int(max_kernel))

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        k = int(np.random.choice(np.arange(3, self.max_kernel + 1, 2)))
        if np.random.rand() < 0.5:
            return cv2.GaussianBlur(image, (k, k), 0), target
        angle = float(np.random.uniform(0, 180))
        return motion_blur(image, k, angle), target


class RandomNoise:
    def __init__(self, p: float = 0.2, sigma: Tuple[float, float] = (3.0, 12.0)):
        self.p, self.sigma = p, tuple(sigma)

    def __call__(self, image, target):
        if np.random.rand() >= self.p:
            return image, target
        sigma = np.random.uniform(*self.sigma)
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        return (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8), target


class RandomOcclusion:
    """Paste random patches over part of some heads; the label stays valid because at
    most ``max_fraction`` of each box is covered."""

    def __init__(self, p: float = 0.3, per_box_p: float = 0.5,
                 fraction: Tuple[float, float] = (0.1, 0.35)):
        self.p, self.per_box_p, self.fraction = p, per_box_p, tuple(fraction)

    def __call__(self, image, target):
        if np.random.rand() >= self.p or len(target["boxes"]) == 0:
            return image, target
        image = image.copy()
        for box in target["boxes"]:
            if np.random.rand() < self.per_box_p:
                occlude_box(image, box, np.random.uniform(*self.fraction), np.random)
        return image, target


# --------------------------------------------------------------------------- helpers
def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    lut = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def motion_blur(image: np.ndarray, k: int, angle: float) -> np.ndarray:
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def occlude_box(image: np.ndarray, box: np.ndarray, fraction: float, rng) -> None:
    """In-place: cover ``fraction`` of the box area with a random-colour rectangle."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return
    aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    area = fraction * bw * bh
    ow = int(round(min(bw, np.sqrt(area * aspect))))
    oh = int(round(min(bh, area / max(ow, 1))))
    if ow < 1 or oh < 1:
        return
    ox = int(x1 + rng.uniform(0, max(bw - ow, 0)))
    oy = int(y1 + rng.uniform(0, max(bh - oh, 0)))
    color = rng.randint(0, 256, size=3) if hasattr(rng, "randint") else rng.integers(0, 256, size=3)
    image[oy:oy + oh, ox:ox + ow] = color.astype(np.uint8)


def build_train_transforms(aug: Optional[dict]) -> Optional[Compose]:
    aug = aug or {}
    if not aug.get("enabled", True):
        return None
    return Compose(
        [
            RandomScaleCrop(aug.get("scale_crop_p", 0.5), aug.get("scale_range", (0.6, 1.0)),
                            aug.get("min_visibility", 0.6)),
            RandomHorizontalFlip(aug.get("hflip_p", 0.5)),
            RandomBrightnessContrast(aug.get("brightness_contrast_p", 0.5),
                                     aug.get("brightness", 0.25), aug.get("contrast", 0.25)),
            RandomGamma(aug.get("gamma_p", 0.3), aug.get("gamma_range", (0.5, 2.2))),
            RandomHSV(aug.get("hsv_p", 0.5)),
            RandomOcclusion(aug.get("occlusion_p", 0.3), aug.get("occlusion_per_box_p", 0.5),
                            aug.get("occlusion_fraction", (0.1, 0.35))),
            RandomBlur(aug.get("blur_p", 0.2)),
            RandomNoise(aug.get("noise_p", 0.2)),
        ]
    )


# --------------------------------------------------------------------------- stress tests
def stress_low_light(image, target, rng):
    dark = apply_gamma(image, 2.5).astype(np.float32) * 0.7
    dark += rng.normal(0, 4.0, image.shape)
    return dark.clip(0, 255).astype(np.uint8), target


def stress_overexposed(image, target, rng):
    bright = apply_gamma(image, 0.45).astype(np.float32) + 30
    return bright.clip(0, 255).astype(np.uint8), target


def stress_occlusion(image, target, rng):
    image = image.copy()
    for box in target["boxes"]:
        occlude_box(image, box, 0.3, rng)
    return image, target


def stress_motion_blur(image, target, rng):
    return motion_blur(image, 9, 0.0), target


STRESS_TESTS = {
    "low_light": stress_low_light,
    "overexposed": stress_overexposed,
    "occlusion_30pct": stress_occlusion,
    "motion_blur": stress_motion_blur,
}
