import numpy as np
import cv2
import pytest

from helmet.dataset import HelmetDataset, LabelMapper, build_dataset, index_dataset
from helmet.transforms import (STRESS_TESTS, RandomHorizontalFlip, RandomScaleCrop, build_train_transforms)

VOC = """<annotation><filename>{f}</filename><size><width>100</width><height>80</height></size>
<object><name>With Helmet</name><occluded>1</occluded><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>
<object><name>Without Helmet</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>120</xmax><ymax>40</ymax></bndbox></object>
<object><name>number plate</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def cfg_for(root, fmt="voc"):
    return {"data": {"root": str(root), "format": fmt, "images_dir": "images", "annotations_dir": "annotations",
                     "labels_dir": "labels", "splits_dir": str(root / "splits"),
                     "classes": ["helmet", "no_helmet"],
                     "aliases": {"helmet": ["with helmet"], "no_helmet": ["without helmet"]},
                     "yolo_names": ["With Helmet", "Without Helmet", "plate"]}}


def make_image(path, w=100, h=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def test_label_mapper_normalises_names():
    m = LabelMapper(["helmet", "no_helmet"], {"no_helmet": ["Without Helmet"]})
    assert m("WITHOUT_HELMET") == 2 and m("without-helmet") == 2 and m("no helmet") == 2
    assert m("Helmet") == 1 and m("plate") is None


def test_voc_dataset(tmp_path):
    make_image(tmp_path / "images" / "a.png")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "a.xml").write_text(VOC.format(f="a.png"))
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "train.txt").write_text("a\n")
    ds = build_dataset(cfg_for(tmp_path), "train")
    img, target = ds[0]
    assert img.shape == (3, 80, 100) and img.max() <= 1.0
    assert target["labels"].tolist() == [1, 2]  # plate ignored
    assert target["boxes"].tolist() == [[10, 10, 30, 30], [50, 20, 100, 40]]  # clipped to width
    _, raw = ds.load(0)
    assert raw["occluded"].tolist() == [True, False]


def test_yolo_dataset(tmp_path):
    make_image(tmp_path / "images" / "b.jpg", 200, 100)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "b.txt").write_text("0 0.5 0.5 0.1 0.2\n1 0.1 0.1 0.3 0.4 0.2 0.3\n2 0.5 0.5 0.1 0.1\n")
    recs = index_dataset(cfg_for(tmp_path, "yolo")["data"])
    ds = HelmetDataset([recs["b"]], ["helmet", "no_helmet"])
    _, t = ds[0]
    assert t["labels"].tolist() == [1, 2]
    np.testing.assert_allclose(t["boxes"][0].numpy(), [90, 40, 110, 60], atol=1e-4)
    np.testing.assert_allclose(t["boxes"][1].numpy(), [20, 10, 60, 40], atol=1e-4)  # polygon -> bbox


def test_horizontal_flip():
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :5] = 255
    t = {"boxes": np.array([[0, 0, 5, 10]], np.float32), "labels": np.array([1])}
    out, tt = RandomHorizontalFlip(p=1.0)(img, t)
    assert tt["boxes"].tolist() == [[15, 0, 20, 10]]
    assert out[:, 15:].min() == 255 and out[:, :15].max() == 0


@pytest.mark.parametrize("seed", range(50))
def test_train_transforms_keep_valid_boxes(seed):
    np.random.seed(seed)
    img = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    boxes = np.array([[10, 10, 40, 40], [100, 50, 150, 110], [60, 5, 70, 15]], np.float32)
    target = {"boxes": boxes, "labels": np.array([1, 2, 1]), "occluded": np.zeros(3, bool),
              "truncated": np.zeros(3, bool)}
    out, t = build_train_transforms({})(img, target)
    h, w = out.shape[:2]
    assert out.dtype == np.uint8 and out.ndim == 3
    assert len(t["boxes"]) == len(t["labels"]) == len(t["occluded"]) >= 1
    b = t["boxes"]
    assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all() and (b[:, 2] <= w).all() and (b[:, 3] <= h).all()
    assert ((b[:, 2] - b[:, 0]) > 0).all() and ((b[:, 3] - b[:, 1]) > 0).all()


def test_scale_crop_never_cuts_heads_below_visibility():
    np.random.seed(0)
    crop = RandomScaleCrop(p=1.0, scale=(0.5, 0.9), min_visibility=0.6)
    boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]], np.float32)
    for _ in range(100):
        img = np.zeros((100, 100, 3), np.uint8)
        _, t = crop(img, {"boxes": boxes.copy(), "labels": np.array([1, 2])})
        areas = (t["boxes"][:, 2] - t["boxes"][:, 0]) * (t["boxes"][:, 3] - t["boxes"][:, 1])
        orig = {1: 1600.0, 2: 900.0}
        for a, l in zip(areas, t["labels"]):
            assert a >= 0.6 * orig[int(l)] - 1e-3


def test_stress_tests_are_deterministic():
    img = np.random.RandomState(0).randint(0, 255, (60, 80, 3)).astype(np.uint8)
    t = {"boxes": np.array([[10, 10, 40, 40]], np.float32), "labels": np.array([1])}
    for fn in STRESS_TESTS.values():
        a, _ = fn(img, t, np.random.default_rng(1))
        b, _ = fn(img, t, np.random.default_rng(1))
        assert np.array_equal(a, b) and a.shape == img.shape
