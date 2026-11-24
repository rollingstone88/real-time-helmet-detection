import math

import numpy as np
import pytest

from helmet.metrics import DetectionEvaluator, average_precision, box_iou, match_predictions

H, NH = 1, 2  # helmet, no_helmet


def pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float),
            "labels": np.array(labels, int)}


def gt(boxes, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "labels": np.array(labels, int)}


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_perfect_predictions():
    ev = DetectionEvaluator(["helmet", "no_helmet"])
    ev.add(pred([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8], [H, NH]), gt([[0, 0, 10, 10], [20, 20, 40, 40]], [H, NH]))
    m = ev.compute(0.5)
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(1.0)
    assert m["precision"] == m["recall"] == m["f1"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_hand_computed_ap_and_prf():
    # GT: two helmets. Preds: 0.9 TP, 0.8 FP, 0.7 TP
    # precision 1, .5, .667 ; recall .5, .5, 1 ; envelope 1, .667, .667
    # 101-pt: recall points 0..0.5 (51 pts) -> 1.0, 0.51..1.0 (50 pts) -> 2/3
    ev = DetectionEvaluator(["helmet", "no_helmet"])
    ev.add(pred([[0, 0, 10, 10], [50, 50, 60, 60], [20, 0, 30, 10]], [0.9, 0.8, 0.7], [H, H, H]),
           gt([[0, 0, 10, 10], [20, 0, 30, 10]], [H, H]))
    m = ev.compute(conf_threshold=0.75)
    assert m["per_class"]["helmet"]["ap50"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert math.isnan(m["per_class"]["no_helmet"]["ap50"])  # no GT -> excluded from mAP
    assert m["map50"] == pytest.approx(m["per_class"]["helmet"]["ap50"])
    c = m["per_class"]["helmet"]
    assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 1)
    assert c["precision"] == pytest.approx(0.5) and c["recall"] == pytest.approx(0.5) and c["f1"] == pytest.approx(0.5)


def test_duplicate_detection_is_false_positive():
    tp = match_predictions(np.array([[0, 0, 10, 10], [0, 0, 10, 10.5]]), np.array([[0, 0, 10, 10]]), [0.5])
    assert tp.tolist() == [[True, False]]


def test_greedy_match_uses_best_unmatched_gt():
    gts = np.array([[0, 0, 10, 10], [2, 0, 12, 10]])
    preds = np.array([[0, 0, 10, 10], [1, 0, 11, 10]])  # 2nd overlaps both; first gt already taken
    assert match_predictions(preds, gts, [0.5]).tolist() == [[True, True]]


def test_iou_threshold_affects_map50_95():
    ev = DetectionEvaluator(["helmet"])
    ev.add(pred([[0, 0, 10, 10]], [0.9], [H]), gt([[0, 0, 10, 13]], [H]))  # IoU = 10/13 = 0.77
    m = ev.compute()
    assert m["map50"] == pytest.approx(1.0)
    # matched at thresholds 0.50..0.75 (6 of 10)
    assert m["map50_95"] == pytest.approx(0.6)


def test_confusion_matrix_class_swap_and_background():
    ev = DetectionEvaluator(["helmet", "no_helmet"])
    ev.add(pred([[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.9], [NH, H]),
           gt([[0, 0, 10, 10], [50, 50, 60, 60]], [H, NH]))
    cm = np.array(ev.compute(0.5)["confusion_matrix"])
    assert cm[0, 1] == 1  # helmet predicted as no_helmet
    assert cm[2, 0] == 1  # false positive helmet on background
    assert cm[1, 2] == 1  # missed no_helmet


def test_average_precision_edge_cases():
    assert math.isnan(average_precision(np.zeros(0, bool), np.zeros(0), 0))
    assert average_precision(np.zeros(0, bool), np.zeros(0), 3) == 0.0
    assert average_precision(np.array([False, True]), np.array([0.2, 0.9]), 1) == pytest.approx(1.0)


def test_best_f1_threshold():
    ev = DetectionEvaluator(["helmet"])
    ev.add(pred([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.3], [H, H]), gt([[0, 0, 10, 10]], [H]))
    t, f1 = ev.best_f1_threshold()
    assert f1 == pytest.approx(1.0) and 0.3 < t <= 0.9
