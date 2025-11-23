# utils/detector.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
def load_detector(weights_path: str | Path) -> YOLO:
    """
    Load YOLOv8 detector from weights path.

    Expects a model trained to detect at least:
      - RBC
      - WBC
      - Platelet

    Returns:
      An Ultralytics YOLO model object.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    return model


# -------------------------------------------------
# IMAGE NORMALIZATION
# -------------------------------------------------
def _ensure_numpy(image: Any) -> np.ndarray:
    """
    Convert various input types to a numpy array in HWC RGB format.
    """
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        # assume HWC; if it's grayscale, expand to 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return image
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"))
    raise TypeError(f"Unsupported image type for detector: {type(image)}")


# -------------------------------------------------
# CLASS NAME NORMALIZATION
# -------------------------------------------------
def _normalize_yolo_label(raw_name: str) -> str:
    """
    Map YOLO class name strings to canonical labels:
        "RBC", "WBC", "Platelet"

    Adjust this logic if your YOLO class names differ.
    """
    name_upper = raw_name.strip().upper()

    # Basic heuristic mapping
    if "RBC" in name_upper or "RED" in name_upper:
        return "RBC"
    if "PLT" in name_upper or "PLATE" in name_upper or "PLATELET" in name_upper:
        return "Platelet"
    # fallback: treat anything else as WBC
    return "WBC"


# -------------------------------------------------
# DETECTION RUNNER
# -------------------------------------------------
def run_detector(
    model: YOLO,
    image: Any,
    imgsz: int = 512,
    conf_thres: float = 0.25,
) -> List[Dict]:
    """
    Run YOLO detection on a single image and return a list of
    detection dicts with normalized labels.

    Returns:
      [
        {
          "box": [x1, y1, x2, y2],
          "label": "RBC" | "WBC" | "Platelet",
          "raw_label": "<original YOLO class name>",
          "score": float
        },
        ...
      ]
    """
    np_img = _ensure_numpy(image)

    results = model.predict(
        source=np_img,
        imgsz=imgsz,
        conf=conf_thres,
        verbose=False,
        device=DEVICE.type if DEVICE.type == "cuda" else "cpu",
    )

    r = results[0]
    det_names = model.names

    detections: List[Dict] = []

    if r.boxes is None or len(r.boxes) == 0:
        return detections

    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        cls_index = int(cls_id.item()) if hasattr(cls_id, "item") else int(cls_id)
        raw_name = str(det_names[cls_index])
        label = _normalize_yolo_label(raw_name)

        x1, y1, x2, y2 = box.tolist()
        score = float(conf.item()) if hasattr(conf, "item") else float(conf)

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "label": label,
                "raw_label": raw_name,
                "score": score,
            }
        )

    return detections
