# utils/analysis.py

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

# -------------------------------------------------
# GLOBALS & LABELS
# -------------------------------------------------

# IMPORTANT: This order must match the training/order used for your WBC classifier.
CLASS_NAMES = [
    "neutrophil",
    "eosinophil",
    "basophil",
    "lymphocyte",
    "monocyte",
    "immature_granulocyte",
    "erythroblast",
    "platelet",
]


# -------------------------------------------------
# CSV RANGE PARSING + REFERENCE HELPERS
# -------------------------------------------------

def _parse_range(txt: str | float | int | None) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a 'low-high' textual range (e.g. '40-70') into (low, high).
    Returns (None, None) if parsing fails.
    """
    if txt is None or pd.isna(txt):
        return (None, None)

    s = str(txt).strip()
    if not s:
        return (None, None)

    s = s.replace("approx.", "")
    parts = [p.strip() for p in s.split("-") if p.strip()]
    if len(parts) < 2:
        return (None, None)

    try:
        low = float(parts[0])
        high = float(parts[1])
        return (low, high)
    except ValueError:
        return (None, None)


def load_reference(
    age_group: str = "Adults (18-60y)",
    gender: Optional[str] = None,
    csv_path: str | Path = "",
) -> Dict:
    """
    Load a single reference row from the WBC differential reference CSV,
    filtered by age group and optionally gender.

    Expected CSV columns (example):
      - Age Group
      - Gender
      - Neutrophils % (Range)
      - Lymphocytes % (Range)
      - Monocytes % (Range)
      - Eosinophils % (Range)
      - Basophils % (Range)
      - Immature Granulocytes %
      - Infection Insights (High)
      - Infection Insights (Low)
    """
    if not csv_path:
        raise ValueError("csv_path must be provided to load_reference().")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    sub = df[df["Age Group"].astype(str).str.strip() == age_group]
    if sub.empty:
        raise ValueError(f"Age group '{age_group}' not found in reference file.")

    if gender:
        g = gender.strip().upper()
        sub2 = sub[sub["Gender"].astype(str).str.strip().str.upper() == g]
        if not sub2.empty:
            sub = sub2

    return sub.iloc[0].to_dict()


# -------------------------------------------------
# AGE/GENDER HELPERS
# -------------------------------------------------

def map_age_to_group(age_years: float) -> str:
    """
    Map a numeric age (in years) to an age-group label
    as described in the WBC reference CSV.
    """
    if age_years < 0.01:   # ~0–3 days
        return "Newborn (0-3d)"
    if age_years < 0.1:    # ~4–28 days
        return "Infant (4-28d)"
    if age_years < 2:      # 1m–2y
        return "Children (1m-2y)"
    if age_years < 6:
        return "Children (2-6y)"
    if age_years < 12:
        return "Children (6-12y)"
    if age_years < 18:
        return "Adolescents (12-18y)"
    if age_years <= 60:
        return "Adults (18-60y)"
    return "Elderly (>60y)"


def pick_gender_for_group(
    age_group: str,
    csv_path: str | Path,
) -> Optional[str]:
    """
    If gender is unknown, pick a valid gender for that age group
    from the reference CSV. Returns 'M', 'F', or None.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    sub = df[df["Age Group"].astype(str).str.strip() == age_group]
    if sub.empty:
        return None

    genders = (
        sub["Gender"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not genders:
        return None

    # If "M/F" is present, just pick randomly
    if any("M/F" in g or "M / F" in g for g in genders):
        return random.choice(["M", "F"])

    return random.choice(genders)


# -------------------------------------------------
# DIFFERENTIAL & REPORT GENERATION
# -------------------------------------------------

def compute_differential_percentages(
    wbc_subtypes: Dict[str, int],
) -> Dict[str, float]:
    """
    Convert WBC subtype counts to percentages.
    Returns a dict with percentages, keyed by subtype name.
    """
    total = sum(wbc_subtypes.values()) if wbc_subtypes else 0
    if total == 0:
        return {k: 0.0 for k in wbc_subtypes.keys()}

    return {
        k: round((v / total) * 100.0, 1)
        for k, v in wbc_subtypes.items()
    }


def generate_report_from_ai(
    ai_result: Dict,
    age_group: str,
    gender: Optional[str],
    csv_path: str | Path,
) -> str:
    """
    Generate a human-readable report text using:
      - AI-derived results (coarse counts & WBC subtypes)
      - Reference ranges from CSV for the given age group & gender.

    ai_result expected keys:
      - patient_id
      - coarse_counts: {"WBC": int, "RBC": int, "Platelet": int}
      - wbc_subtypes: {subtype_name: int}
      - fovs_analyzed (optional)
      - calibration (optional, dict with FOV area, constant)
      - timestamp (optional)
    """

    ref = load_reference(age_group=age_group, gender=gender, csv_path=csv_path)

    patient_id = ai_result.get("patient_id", "UNKNOWN")
    ts = ai_result.get("timestamp") or datetime.now().isoformat(timespec="seconds")

    fovs = ai_result.get("fovs_analyzed", 0)
    coarse = ai_result.get("coarse_counts", {}) or {}
    subtypes = ai_result.get("wbc_subtypes", {}) or {}
    calib = ai_result.get("calibration", {}) or {}

    fov_area = calib.get("fov_area_mm2")
    calib_const = calib.get("calibration_constant")

    total_wbc = coarse.get("WBC", 0)
    total_rbc = coarse.get("RBC", 0)
    total_plt = coarse.get("Platelet", 0)

    classified_total = sum(subtypes.values()) if subtypes else 0

    def ai_pct(name: str) -> float:
        if not classified_total:
            return 0.0
        return round((subtypes.get(name, 0) / classified_total) * 100.0, 1)

    ai_neut = ai_pct("neutrophil")
    ai_lymph = ai_pct("lymphocyte")
    ai_mono = ai_pct("monocyte")
    ai_eos = ai_pct("eosinophil")
    ai_baso = ai_pct("basophil")
    ai_ig = ai_pct("immature_granulocyte")
    ai_ery = ai_pct("erythroblast")

    # parse reference ranges from CSV columns
    ref_neut_lo, ref_neut_hi = _parse_range(ref.get("Neutrophils % (Range)"))
    ref_lymph_lo, ref_lymph_hi = _parse_range(ref.get("Lymphocytes % (Range)"))
    ref_mono_lo, ref_mono_hi = _parse_range(ref.get("Monocytes % (Range)"))
    ref_eos_lo, ref_eos_hi = _parse_range(ref.get("Eosinophils % (Range)"))
    ref_baso_lo, ref_baso_hi = _parse_range(ref.get("Basophils % (Range)"))

    ref_ig_txt = str(ref.get("Immature Granulocytes %", "")).lower()
    # crude check — if "3" is mentioned, set 3% as an upper threshold
    ref_ig_max = 3.0 if "3" in ref_ig_txt else None

    high_note = ref.get("Infection Insights (High)", "")
    low_note = ref.get("Infection Insights (Low)", "")

    insights: list[str] = []

    def check_range(label: str, ai_val: float, lo: Optional[float], hi: Optional[float]):
        if lo is None or hi is None:
            return
        # Above reference range
        if ai_val > hi and high_note:
            insights.append(
                f"- {label} {ai_val}% is above reference ({lo}-{hi}%). {high_note}"
            )
        # Below reference range
        elif ai_val < lo and low_note:
            insights.append(
                f"- {label} {ai_val}% is below reference ({lo}-{hi}%). {low_note}"
            )

    check_range("Neutrophils", ai_neut, ref_neut_lo, ref_neut_hi)
    check_range("Lymphocytes", ai_lymph, ref_lymph_lo, ref_lymph_hi)
    check_range("Monocytes", ai_mono, ref_mono_lo, ref_mono_hi)
    check_range("Eosinophils", ai_eos, ref_eos_lo, ref_eos_hi)
    check_range("Basophils", ai_baso, ref_baso_lo, ref_baso_hi)

    if ref_ig_max is not None and ai_ig > ref_ig_max:
        insights.append(
            f"- Immature granulocytes {ai_ig}% > allowed ({ref_ig_max}%), "
            "suggesting left shift or active marrow response. Recommend manual review."
        )

    if ai_ery > 0:
        insights.append(
            f"- Erythroblasts detected ({ai_ery}%), unusual in normal peripheral blood → manual review recommended."
        )

    if classified_total < 100:
        insights.append(
            f"- Only {classified_total} WBCs classified; differential may be statistically unstable. "
            "Consider reviewing more fields."
        )

    if not fov_area or not calib_const:
        insights.append(
            "- Absolute counts per µL are not reported (FOV area and calibration "
            "constant not provided). Use results as qualitative screening."
        )

    # Assemble multi-line report
    lines: list[str] = []
    lines.append("AI-Assisted Peripheral Blood Smear Report (Prototype)")
    lines.append("================================================================")
    lines.append(f"Patient ID       : {patient_id}")
    lines.append(f"Date/Time        : {ts}")
    lines.append(f"Age Group (ref)  : {age_group}")
    if gender:
        lines.append(f"Gender (ref)     : {gender}")
    if fovs:
        lines.append(f"FOVs Analyzed    : {fovs}")
    lines.append("")

    lines.append("1. Coarse Counts (sum over analyzed fields)")
    lines.append(f"   WBC       : {total_wbc}")
    lines.append(f"   RBC       : {total_rbc}")
    lines.append(f"   Platelets : {total_plt}")
    lines.append("")

    lines.append("2. AI Differential vs Reference")
    lines.append(f"   Neutrophils       : {ai_neut}%   (ref {ref.get('Neutrophils % (Range)')})")
    lines.append(f"   Lymphocytes       : {ai_lymph}%  (ref {ref.get('Lymphocytes % (Range)')})")
    lines.append(f"   Monocytes         : {ai_mono}%   (ref {ref.get('Monocytes % (Range)')})")
    lines.append(f"   Eosinophils       : {ai_eos}%    (ref {ref.get('Eosinophils % (Range)')})")
    lines.append(f"   Basophils         : {ai_baso}%   (ref {ref.get('Basophils % (Range)')})")
    lines.append(f"   Imm. granulocytes : {ai_ig}%     (ref {ref.get('Immature Granulocytes %')})")
    lines.append(f"   Erythroblasts     : {ai_ery}%     (no reference range in file)")
    lines.append("")

    lines.append("3. Calibration")
    lines.append(f"   FOV area (mm²)         : {fov_area if fov_area else 'not provided'}")
    lines.append(f"   Calibration constant   : {calib_const if calib_const else 'not provided'}")
    lines.append("")

    lines.append("4. AI Insights")
    if insights:
        for msg in insights:
            lines.append(f"   {msg}")
    else:
        lines.append("   All AI-derived percentages fall within reference ranges for this age group.")
    lines.append("")
    lines.append(
        "Method: YOLO-based detector (RBC/WBC/Platelet) + WBC subtype classifier "
        "compared against age-/gender-specific reference ranges from CSV.\n"
        "This is a research prototype and not a substitute for formal lab testing."
    )

    return "\n".join(lines)
