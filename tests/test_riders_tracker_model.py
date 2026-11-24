import numpy as np
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from helmet.model import build_detector, load_helmet_model, save_checkpoint
from helmet.riders import associate_heads, find_riders, is_riding
from helmet.tracker import IoUTracker


def test_is_riding():
    moto = [100, 200, 200, 300]
    assert is_riding([120, 80, 180, 260], moto)          # sitting on the bike
    assert not is_riding([300, 80, 360, 260], moto)      # beside the bike
    assert not is_riding([120, 260, 180, 500], moto)     # far below (pedestrian in front)
    # measured on a real photo: pedestrian standing behind a large parked motorcycle
    assert not is_riding([464, 233, 539, 444], [29, 324, 617, 613])
    assert is_riding([184, 239, 365, 626], [29, 324, 617, 613])      # the actual rider
    assert find_riders(np.array([[120, 80, 180, 260], [300, 80, 360, 260]]), np.array([moto])).tolist() == [0]


def test_associate_heads_two_riders_one_bike():
    riders = np.array([[100, 50, 160, 250], [140, 70, 200, 260]])   # driver + pillion overlap
    heads = np.array([[150, 72, 175, 97], [115, 52, 140, 77], [0, 0, 10, 10]])
    assign = associate_heads(riders, heads)
    assert assign == {0: 1, 1: 0}  # each rider gets its own head; stray head ignored


def test_head_must_be_in_upper_part():
    riders = np.array([[0, 0, 100, 200]])
    assert associate_heads(riders, np.array([[40, 150, 60, 170]])) == {}


def test_tracker_keeps_ids_and_smooths_status():
    tr = IoUTracker(iou_threshold=0.3, max_missed=2, history=5, min_hits_for_violation=3)
    seq = ["no_helmet", "no_helmet", "unknown", "helmet", "no_helmet"]
    ids = []
    for i, s in enumerate(seq):
        (tid, status), = tr.update([[10 + i, 10, 60 + i, 110]], [s])
        ids.append(tid)
    assert len(set(ids)) == 1
    assert status == "no_helmet"
    assert tr.violators == {ids[0]}
    # disappears longer than max_missed -> new id
    for _ in range(3):
        tr.update([], [])
    (tid, _), = tr.update([[10, 10, 60, 110]], ["helmet"])
    assert tid != ids[0]


def test_checkpoint_roundtrip_with_frozen_bn(tmp_path):
    torch.manual_seed(0)
    cfg = {"model": {"arch": "fasterrcnn_mobilenet_v3_large_fpn", "min_size": 128, "max_size": 160},
           "data": {"classes": ["helmet", "no_helmet"]}}
    model = build_detector(cfg["model"]["arch"], num_classes=3, pretrained="none", min_size=128, max_size=160)
    # simulate a pretrained backbone, which torchvision builds with FrozenBatchNorm2d
    bn_name, bn = next((n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d))
    parent = model.get_submodule(bn_name.rpartition(".")[0])
    frozen = FrozenBatchNorm2d(bn.num_features)
    frozen.running_mean.uniform_(-0.1, 0.1)
    setattr(parent, bn_name.rpartition(".")[2], frozen)
    model.eval()
    save_checkpoint(tmp_path / "m.pt", model, cfg, epoch=1, metrics={"map50": np.float64(0.5)})
    loaded, ckpt = load_helmet_model(tmp_path / "m.pt", torch.device("cpu"))
    assert isinstance(loaded.get_submodule(bn_name), FrozenBatchNorm2d)
    assert ckpt["metrics"]["map50"] == 0.5 and ckpt["class_names"] == ["helmet", "no_helmet"]
    x = [torch.rand(3, 120, 150)]
    with torch.no_grad():
        a = model.backbone(model.transform(x)[0].tensors)
        b = loaded.backbone(loaded.transform(x)[0].tensors)
    for k in a:
        assert torch.allclose(a[k], b[k])
