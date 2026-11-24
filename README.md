# Real-Time Helmet Detection from Videos

**Python · PyTorch · OpenCV**

A computer-vision system that detects **motorcycle riders** in recorded or live video and
identifies whether each rider is **wearing a helmet**.

<p align="center"><img src="assets/demo.gif" alt="Helmet detection on video: tracked riders, helmet / no-helmet heads with confidence scores" width="560"></p>

**Test-set results** (115 unseen images, Kaggle T4 GPU):

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Speed |
|---|---|---|---|---|---|---|
| Faster R-CNN MobileNetV3-FPN (real-time) | 0.811 | 0.495 | 0.776 | 0.822 | 0.798 | 27 ms/img (~37 FPS) |
| Faster R-CNN ResNet-50-FPN v2 (accuracy) | 0.846 | 0.511 | 0.690 | 0.913 | 0.786 | 175 ms/img |

Details in [Results](#results). The whole pipeline is also available as a single notebook: [helmet_detection_pipeline.ipynb](helmet_detection_pipeline.ipynb).

```
video frame ──► OpenCV ──► Faster R-CNN (fine-tuned) ──► helmet / no_helmet head boxes ─┐
 (file, webcam,             transfer learning from COCO                                 ├─► head ↔ rider ──► IoU tracker ──► annotated
  RTSP stream)  └─────────► Faster R-CNN (COCO) ──► person-on-motorcycle = rider ──────┘    association       (IDs, smoothed   frame + stats
                                                                                                              status)
```

## What is implemented

| Component | Where |
|---|---|
| Detect riders and helmet status in recorded **or** real-time video (file / webcam / RTSP) | [infer.py](infer.py), [helmet/pipeline.py](helmet/pipeline.py), [helmet/riders.py](helmet/riders.py) |
| PyTorch object detection, **transfer learning** from a COCO-pretrained Faster R-CNN (new 3-class head, partially frozen backbone) | [helmet/model.py](helmet/model.py) |
| Custom **preprocessing** (VOC/YOLO parsing, class-name mapping, box cleaning, stratified splits) | [helmet/dataset.py](helmet/dataset.py), [prepare_data.py](prepare_data.py) |
| Custom box-aware **augmentation** (lighting, gamma, occlusion, blur, noise, zoom-crop, flip) | [helmet/transforms.py](helmet/transforms.py) |
| **Training / validation** loops (warm-up + cosine LR, grad clipping, AMP on CUDA, best-checkpointing, early stopping, resume) | [train.py](train.py), [helmet/engine.py](helmet/engine.py) |
| **Inference** pipeline with OpenCV visualisation: bounding boxes, class labels, confidence scores, FPS, rider counts | [helmet/visualize.py](helmet/visualize.py) |
| **Precision, recall, F1, mAP@0.5, mAP@0.5:0.95**, PR / F1 curves, confusion matrix (implemented from scratch, unit-tested) | [helmet/metrics.py](helmet/metrics.py), [evaluate.py](evaluate.py) |
| Handling **occlusion, varying lighting, multiple riders** in training, inference and evaluation | see [Robustness](#robustness-occlusion-lighting-multiple-riders) |

## Setup

```bash
git clone https://github.com/rollingstone88/real-time-helmet-detection.git
cd real-time-helmet-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # 70 unit tests
```

The first training / inference run downloads the COCO-pretrained weights from
`download.pytorch.org` (~75 MB for MobileNetV3, ~170 MB for ResNet-50).

## 1. Get a dataset

Any helmet dataset in **Pascal VOC** or **YOLO** format works. Class names are mapped onto the
project classes `helmet` / `no_helmet` through `data.aliases` in
[configs/default.yaml](configs/default.yaml) (e.g. `With Helmet`, `Without Helmet`, `head`, …).
Classes that are not mapped (e.g. `number plate`) are ignored.

**Option A: Kaggle "Helmet Detection" (Pascal VOC, 764 images, `With Helmet` / `Without Helmet`)**
<https://www.kaggle.com/datasets/andrewmvd/helmet-detection>

```
data/helmet/images/BikesHelmets0.png ...
data/helmet/annotations/BikesHelmets0.xml ...
```
```bash
python prepare_data.py --config configs/default.yaml          # 70 / 15 / 15 stratified split
```

**Option B: a Roboflow / YOLO export** with `train/ valid/ test/` folders:

```bash
python prepare_data.py --config configs/default.yaml --from-presplit path/to/export \
  --set data.format=yolo data.root=data/helmet_yolo data.splits_dir=data/helmet_yolo/splits \
        "data.yolo_names=[With Helmet, Without Helmet]"      # names in YOLO id order (from data.yaml)
```
Pass the same `--set` overrides to `train.py` (they are saved into the checkpoint afterwards).

## 2. Train

```bash
python train.py --config configs/default.yaml
# more accurate, slower:
python train.py --config configs/default.yaml --set model.arch=fasterrcnn_resnet50_fpn_v2 train.batch_size=2
# continue an interrupted run:
python train.py --resume runs/exp/last.pt
```

Each run goes to `runs/exp`, `runs/exp2`, … and contains `best.pt`, `last.pt`, `history.csv`,
`training_curves.png`, `val_metrics_best.json` and the exact `config.yaml`. Validation
(mAP, precision, recall, F1) runs after every epoch; `best.pt` is selected by
`train.best_metric` (default mAP@0.5).

## 3. Evaluate

```bash
python evaluate.py --weights runs/exp/best.pt --stress
```

Prints and saves (`runs/exp/eval_test/`):

* per-class and overall **AP@0.5, AP@0.5:0.95, precision, recall, F1**, mAP@0.75, macro-F1
* the confidence threshold with the best F1 (use it as `--conf` at inference)
* **confusion matrix** (helmet ↔ no_helmet swaps, false alarms, misses)
* breakdown by **lighting** (dark / normal / bright), **single vs multiple riders**, and
  **annotated occlusion / truncation**
* **stress tests** (`--stress`): low light, over-exposure, 30 % occlusion of every head,
  motion blur, with the mAP drop versus clean images
* inference speed (ms / image, FPS)
* `report.json`, `pr_curve.png`, `f1_curve.png`, `confusion_matrix.png`

Metric definitions (COCO protocol): predictions are matched greedily by score to the
unmatched ground-truth box with the highest IoU; AP is 101-point interpolated; precision /
recall / F1 are computed at IoU 0.5 for detections above the confidence threshold.

## 4. Run on video

```bash
python infer.py --weights runs/exp/best.pt --source traffic.mp4 --show       # recorded video
python infer.py --weights runs/exp/best.pt --source 0 --show                  # webcam (real time)
python infer.py --weights runs/exp/best.pt --source rtsp://user:pass@cam/stream --show
python infer.py --weights runs/exp/best.pt --source photos/                   # images
```

Useful flags: `--conf 0.5`, `--enhance-low-light`, `--every 2` (detect every 2nd frame),
`--no-riders` (heads only, faster), `--no-track`, `--csv detections.csv`, `--output out.mp4`.
Keys: `q`/`Esc` quit, `p` pause.

Each frame shows:
* head boxes labelled `helmet 0.94` / `no_helmet 0.88` (green / red)
* rider boxes labelled `Rider #3: NO HELMET` (track ID + smoothed status, grey = unknown)
* a panel with FPS, head counts, rider counts and the number of unique riders without a helmet

### How riders are identified

1. The fine-tuned model detects heads as `helmet` / `no_helmet`. A class-agnostic NMS keeps
   one label per head.
2. A COCO-pretrained Faster R-CNN detects `person` and `motorcycle`. A person is a **rider**
   when they overlap the motorcycle horizontally, their upper body is above it and their lower
   body reaches into it ([helmet/riders.py](helmet/riders.py)). Pedestrians are excluded.
3. Heads are assigned to riders one-to-one (head mostly inside the rider box, in its upper
   half). This handles a driver and a pillion passenger on one bike, as well as several bikes.
4. An IoU tracker gives riders stable IDs and takes a majority vote of each rider's status
   over the last 15 frames. A rider only counts as a violation after 3 consistent frames.

## Notebook: the whole pipeline in one file

[helmet_detection_pipeline.ipynb](helmet_detection_pipeline.ipynb) re-implements the complete
system end to end, without importing the `helmet` package:

1. dataset parsing and exploration;
2. stratified split;
3. box-aware augmentation;
4. transfer learning;
5. metrics (with sanity checks);
6. training and validation;
7. test evaluation with stress tests;
8. rider detection and tracking;
9. OpenCV inference on images, a generated demo video and live webcam / RTSP streams.

The checkpoints it saves can be used directly with `infer.py`.

* **Kaggle:** add the dataset `andrewmvd/helmet-detection`, choose a GPU accelerator and turn
  internet on, then *Run All* (about 1 hour on a T4 for both models).
* **Locally:** run it from the repository root with the dataset in `data/helmet/`. Set
  `FAST_DEV_RUN = True` for a 5–10 minute smoke test.

## Results

Kaggle "Helmet Detection" dataset (764 images, 1,451 heads). The stratified split is
534 / 115 / 115 images. The models were trained on a Kaggle T4 GPU with the notebook,
and the best checkpoint was selected on validation mAP@0.5. All numbers below are on the test split.

| | MobileNetV3-FPN | ResNet-50-FPN v2 |
|---|---|---|
| mAP@0.5 | 0.811 | **0.846** |
| mAP@0.5:0.95 | 0.495 | **0.511** |
| Precision / Recall / F1 (conf 0.5) | **0.776** / 0.822 / **0.798** | 0.690 / **0.913** / 0.786 |
| AP@0.5 helmet / no_helmet | 0.869 / 0.753 | 0.899 / 0.793 |
| Best-F1 confidence threshold | 0.56 | 0.69 |
| Inference (detector, batch 1, T4) | **27 ms** | 175 ms |
| Training time (early-stopped) | 11.7 min | 33.9 min |

**Robustness** (mAP@0.5 on the test set with synthetic conditions):

| Condition | MobileNetV3-FPN | ResNet-50-FPN v2 |
|---|---|---|
| clean | 0.811 | 0.846 |
| low light | 0.781 | 0.803 |
| over-exposed | 0.803 | 0.831 |
| 30 % of every head occluded | 0.772 | 0.813 |
| motion blur | 0.729 | 0.696 |
| *images with a single rider / multiple riders* | *0.896 / 0.777* | *0.859 / 0.847* |

Notes:
* The test set is small (115 images), so expect differences of ±0.02 between training runs.
* The dataset has very few very dark or very bright images (2 and 5 in the test set). The
  synthetic stress tests are therefore the meaningful measure of robustness to lighting.
* `no_helmet` is the weaker class. It has half as many examples as `helmet`, and most of its
  errors are false alarms on background.

<p align="center"><img src="assets/predictions.jpg" alt="Annotated test images: riders, helmet and no-helmet heads"></p>
<p align="center"><img src="assets/evaluation_mobilenet_v3.png" alt="PR curves, F1 vs confidence, confusion matrix"></p>
<p align="center"><img src="assets/training_curves_mobilenet_v3.png" alt="Training curves"></p>
<p align="center"><img src="assets/augmentations.jpg" alt="Training augmentations and evaluation stress tests"></p>

## Robustness (occlusion, lighting, multiple riders)

| Challenge | Training | Inference | Evaluation |
|---|---|---|---|
| **Occlusion** | `RandomOcclusion` pastes patches over 10–35 % of heads; zoom-crops never keep a cut-off head | tracker keeps a rider's ID for up to 15 missed frames and smooths the status | annotated-occlusion subset + 30 % occlusion stress test |
| **Lighting** | brightness/contrast, gamma 0.5–2.2 (night ↔ glare), HSV jitter, sensor noise | optional CLAHE enhancement for dark frames (`--enhance-low-light`) | dark / normal / bright subsets + low-light and over-exposure stress tests |
| **Multiple riders** | multi-object images; class-stratified splits; optional balanced sampling | one-to-one head↔rider assignment, per-rider tracking IDs, cross-class NMS | single- vs multi-rider subsets |
| **Motion / small objects** | Gaussian + motion blur, zoom-crops (scale 0.6–1.0), FPN multi-scale features | `--every N` to keep up with the frame rate | motion-blur stress test |

## Models

| `model.arch` | Speed | Accuracy | Use |
|---|---|---|---|
| `fasterrcnn_mobilenet_v3_large_fpn` (default) | fast | good | real-time video |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | fastest | lower (small heads) | low-power CPUs |
| `fasterrcnn_resnet50_fpn_v2` | slow | best | offline analysis / when a GPU is available |

**Devices:** `--device auto` uses CUDA if available, otherwise the CPU. Apple-Silicon `mps` is
opt-in only. With torch 2.8, torchvision's Faster R-CNN on MPS gave exploding losses and stalled
during training, and it ran slower than on the CPU.

## Pipeline check with synthetic data

To check that everything runs end-to-end before downloading a real dataset (cartoon images; **not** a real model):

```bash
python tools/make_synthetic_dataset.py --out data/synthetic
python prepare_data.py --set data.root=data/synthetic data.splits_dir=data/synthetic/splits
python train.py --set data.root=data/synthetic data.splits_dir=data/synthetic/splits train.epochs=8 train.output_dir=runs/synthetic
python evaluate.py --weights runs/synthetic/best.pt --stress
python infer.py --weights runs/synthetic/best.pt --source data/synthetic/test_video.mp4 --no-riders
```

## Project layout

```
real-time-helmet-detection/
├── helmet_detection_pipeline.ipynb   self-contained end-to-end notebook
├── configs/default.yaml        all settings (data, model, training, augmentation, inference)
├── prepare_data.py             stratified train/val/test splits, YOLO export import
├── train.py                    transfer-learning training with per-epoch validation
├── evaluate.py                 metrics, breakdowns, stress tests, plots
├── infer.py                    video / webcam / stream / image inference
├── helmet/
│   ├── dataset.py              VOC + YOLO loading, class aliases, box cleaning
│   ├── transforms.py           box-aware augmentations + stress-test transforms
│   ├── model.py                Faster R-CNN builders, head replacement, checkpoints
│   ├── engine.py               train / predict loops, LR schedule
│   ├── metrics.py              IoU matching, AP, mAP, P/R/F1, confusion matrix
│   ├── riders.py               person-on-motorcycle detection, head↔rider assignment
│   ├── tracker.py              IoU tracker with status smoothing
│   ├── pipeline.py             per-frame inference
│   ├── visualize.py            OpenCV drawing
│   └── plots.py                training / evaluation figures
├── tools/make_synthetic_dataset.py
├── assets/                     README figures
└── tests/                      unit tests (metrics, data, transforms, riders, tracker, checkpoints)
```

The dataset is not included. Download it from
[Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection).
