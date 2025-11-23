# utils/report.py

from __future__ import annotations
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image

from utils.analysis import (
    compute_differential_percentages,
    generate_report_from_ai,
)


# -------------------------------------------------
# OPTIONAL IMAGE ENCODING (for API responses)
# -------------------------------------------------
def pil_to_base64(img: Image.Image, format="PNG") -> str:
    """
    Convert a PIL image to Base64 (useful for overlay images in API mode).
    """
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


# -------------------------------------------------
# STRUCTURED API RESPONSE GENERATION
# -------------------------------------------------
def build_api_response(
    ai_result: Dict[str, Any],
    age_group: str,
    gender: Optional[str],
    reference_csv: str,
    overlay_image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Build a clean JSON response combining:
      - coarse counts (RBC/WBC/Platelets)
      - WBC subtype counts
      - WBC subtype percentages
      - generated plain-text report
      - optional overlay image (Base64)
    """

    coarse = ai_result.get("coarse_counts", {})
    subtypes = ai_result.get("wbc_subtypes", {})

    pct = compute_differential_percentages(subtypes)

    # Generate full text report (human readable)
    text_report = generate_report_from_ai(
        ai_result=ai_result,
        age_group=age_group,
        gender=gender,
        csv_path=reference_csv,
    )

    # Build JSON response
    response = {
        "patient_id": ai_result.get("patient_id"),
        "timestamp": ai_result.get("timestamp"),
        "fovs_analyzed": ai_result.get("fovs_analyzed"),
        "coarse_counts": coarse,
        "wbc_subtypes": subtypes,
        "wbc_percentages": pct,
        "report_text": text_report,
        "calibration": ai_result.get("calibration", {}),
    }

    # Add overlay image only if provided
    if overlay_image is not None:
        response["overlay_image"] = pil_to_base64(overlay_image)

    return response


# -------------------------------------------------
# SIMPLE TEXT-ONLY REPORT (UI mode)
# -------------------------------------------------
def build_text_report(
    ai_result: Dict[str, Any],
    age_group: str,
    gender: Optional[str],
    reference_csv: str,
) -> str:
    """
    Helper to get only the text report.
    Useful for Gradio or CLI mode.
    """
    return generate_report_from_ai(
        ai_result=ai_result,
        age_group=age_group,
        gender=gender,
        csv_path=reference_csv,
    )
