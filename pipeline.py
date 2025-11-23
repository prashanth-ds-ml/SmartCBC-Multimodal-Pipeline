# pipeline.py

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List

import numpy as np
import torch
from PIL import Image

from utils.detector import load_detector, run_detector
from utils.classifier import load_wbc_classifier, classify_wbc_crop
from utils.analysis import CLASS_NAMES, map_age_to_group, pick_gender_for_group
from utils.report import build_api_response

# -------------------------------------------------
# PATHS & DEVICE
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

DETECTOR_WEIGHTS = REPO_ROOT / "yolov8_detector" / "best.pt"
CLASSIFIER_WEIGHTS = REPO_ROOT / "wbc_classifier" / "best_model_checkpoint.pth"
REF_CSV = REPO_ROOT / "data" / "WBC differential references.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmartCBC:
    """
    Main SmartCBC pipeline orchestrator.

    Usage (single FOV):

        from pipeline import SmartCBC
        cbc = SmartCBC()
        result = cbc.analyze(image, age=32, gender="M")

    Usage (multiple FOVs):

        result = cbc.analyze_batch([img1, img2, img3], age=32, gender="M")

    You can also pass a list directly to `analyze()` and it will auto-route:

        result = cbc.analyze([img1, img2, img3], age=32, gender="M")

    `result` is a dict ready for API / UI usage:
        - patient_id
        - timestamp
        - fovs_analyzed
        - coarse_counts         (RBC/WBC/Platelet)
        - wbc_subtypes          (raw counts per subtype)
        - wbc_percentages       (percent per subtype)
        - report_text           (plain text report)
        - calibration           (placeholders for now)
    """

    def __init__(
        self,
        detector_weights: Optional[str | Path] = None,
        classifier_weights: Optional[str | Path] = None,
        ref_csv: Optional[str | Path] = None,
        conf_thres: float = 0.25,
        imgsz: int = 512,
    ) -> None:
        self.detector_weights = str(detector_weights or DETECTOR_WEIGHTS)
        self.classifier_weights = str(classifier_weights or CLASSIFIER_WEIGHTS)
        self.ref_csv = str(ref_csv or REF_CSV)

        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.device = DEVICE

        # Load models once
        self.detector = load_detector(self.detector_weights)
        self.classifier = load_wbc_classifier(self.classifier_weights)

    # -------------------------------------------------
    # PUBLIC ENTRYPOINT (single OR multi)
    # -------------------------------------------------
    def analyze(
        self,
        image: Any | Sequence[Any],
        age: Optional[float] = None,
        gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SmartCBC analysis.

        If `image` is:
          - a single image (PIL / np.ndarray / path) -> analyze one FOV
          - a list/tuple of images                      -> aggregate over multiple FOVs
        """

        # If multiple FOVs are provided, delegate to analyze_batch()
        if isinstance(image, (list, tuple)):
            return self.analyze_batch(list(image), age=age, gender=gender)

        # Single-image path (current behavior)
        pil_img = self._ensure_pil(image)

        age_years, age_group, gender = self._resolve_age_gender(age, gender)

        coarse_counts, subtype_counts = self._run_models_on_image(pil_img)

        ai_result = {
            "patient_id": f"PAT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fovs_analyzed": 1,
            "coarse_counts": coarse_counts,
            "wbc_subtypes": subtype_counts,
            "calibration": {
                "fov_area_mm2": None,
                "calibration_constant": None,
            },
        }

        response = build_api_response(
            ai_result=ai_result,
            age_group=age_group,
            gender=gender,
            reference_csv=self.ref_csv,
            overlay_image=None,
        )

        response["age_years"] = age_years
        response["age_group"] = age_group
        response["gender"] = gender

        return response

    # -------------------------------------------------
    # NEW: MULTI-FOV ANALYSIS
    # -------------------------------------------------
    def analyze_batch(
        self,
        images: Sequence[Any],
        age: Optional[float] = None,
        gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SmartCBC analysis over MULTIPLE FOV images.

        Parameters
        ----------
        images : list/tuple of PIL / np.ndarray / path
        age    : float (years), optional
        gender : "M" | "F", optional

        Returns
        -------
        Aggregated result dict:
            - fovs_analyzed = len(images)
            - coarse_counts (sum over all FOVs)
            - wbc_subtypes (sum over all FOVs)
            - wbc_percentages, report_text, etc.
        """

        if not images:
            raise ValueError("analyze_batch() received an empty images list.")

        age_years, age_group, gender = self._resolve_age_gender(age, gender)

        # Initialize aggregate counts
        agg_coarse: Dict[str, int] = {"WBC": 0, "RBC": 0, "Platelet": 0}
        agg_subtypes: Dict[str, int] = {name: 0 for name in CLASS_NAMES}

        fov_count = 0

        for img in images:
            pil_img = self._ensure_pil(img)
            coarse_counts, subtype_counts = self._run_models_on_image(pil_img)

            # Aggregate coarse counts
            for k, v in coarse_counts.items():
                agg_coarse[k] = agg_coarse.get(k, 0) + v

            # Aggregate subtype counts
            for k, v in subtype_counts.items():
                agg_subtypes[k] = agg_subtypes.get(k, 0) + v

            fov_count += 1

        ai_result = {
            "patient_id": f"PAT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fovs_analyzed": fov_count,
            "coarse_counts": agg_coarse,
            "wbc_subtypes": agg_subtypes,
            "calibration": {
                "fov_area_mm2": None,
                "calibration_constant": None,
            },
        }

        response = build_api_response(
            ai_result=ai_result,
            age_group=age_group,
            gender=gender,
            reference_csv=self.ref_csv,
            overlay_image=None,
        )

        response["age_years"] = age_years
        response["age_group"] = age_group
        response["gender"] = gender

        return response

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------
    def _resolve_age_gender(
        self,
        age: Optional[float],
        gender: Optional[str],
    ) -> tuple[float, str, Optional[str]]:
        """
        Compute age_years, age_group, gender with defaults and CSV-based inference.
        """
        age_years = float(age) if age is not None else 30.0
        age_group = map_age_to_group(age_years)

        if gender is None or str(gender).strip() == "":
            gender = pick_gender_for_group(age_group, csv_path=self.ref_csv)
        gender = None if gender is None else str(gender).upper()

        return age_years, age_group, gender

    def _ensure_pil(self, image: Any) -> Image.Image:
        """
        Convert various input types to a PIL.Image in RGB mode.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _run_models_on_image(
        self,
        img: Image.Image,
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Run YOLO detector + WBC classifier on a single image.

        Returns
        -------
        coarse_counts : {"WBC": int, "RBC": int, "Platelet": int}
        subtype_counts: {subtype_name: int}
        """
        coarse_counts: Dict[str, int] = {"WBC": 0, "RBC": 0, "Platelet": 0}
        subtype_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}

        detections = run_detector(
            model=self.detector,
            image=img,
            imgsz=self.imgsz,
            conf_thres=self.conf_thres,
        )

        w, h = img.size

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]  # "RBC", "WBC", "Platelet"

            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)
            x2 = min(int(x2), w)
            y2 = min(int(y2), h)

            if label in coarse_counts:
                coarse_counts[label] += 1
            else:
                coarse_counts[label] = 1

            if label == "WBC" and x2 > x1 and y2 > y1:
                crop = img.crop((x1, y1, x2, y2))
                subtype = classify_wbc_crop(self.classifier, crop)
                if subtype in subtype_counts:
                    subtype_counts[subtype] += 1
                else:
                    subtype_counts[subtype] = 1

        return coarse_counts, subtype_counts
