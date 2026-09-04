# SmartCBC — Multimodal AI Blood Smear Analysis Pipeline

AI-assisted peripheral blood smear analysis: automated cell detection, WBC subtype classification, and age-/gender-specific differential reporting — packaged as a research prototype, not a diagnostic tool.

**Live demo:** [huggingface.co/spaces/KPrashanth/SmartCBC](https://huggingface.co/spaces/KPrashanth/SmartCBC)

## What it does

Given one or more peripheral blood smear field-of-view (FOV) images, SmartCBC:

1. Detects RBCs, WBCs, and Platelets with a YOLOv8 model.
2. Classifies each detected WBC into one of 8 subtypes with a ResNet50 classifier.
3. Aggregates counts across all uploaded FOVs (multi-FOV support).
4. Compares the resulting differential against age-/gender-specific reference ranges.
5. Produces a human-readable report and/or structured JSON, with basic clinical insights (e.g. left-shift flags, erythroblast presence, small-sample warnings).

## Architecture

```
app.py              Gradio UI entrypoint — loads SmartCBC() once at startup
pipeline.py          SmartCBC orchestrator: analyze() / analyze_batch()
utils/detector.py     YOLOv8 detector (RBC / WBC / Platelet)
utils/classifier.py   ResNet50 WBC subtype classifier (8-class head)
utils/analysis.py     Age→age-group mapping, CSV reference lookup, differential %, insights
utils/report.py       Assembles the final JSON/text response
data/                 WBC differential reference ranges (CSV)
yolov8_detector/      YOLOv8 detector weights (best.pt)
wbc_classifier/       ResNet50 classifier checkpoint (best_model_checkpoint.pth, Git LFS)
```

WBC subtype classes (fixed order, must match classifier training):
`neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature_granulocyte, erythroblast, platelet`

## Setup

```bash
pip install -r requirements.txt
```

Requires the two model weight files to be present at:
- `yolov8_detector/best.pt`
- `wbc_classifier/best_model_checkpoint.pth`

## Usage

**Gradio UI:**
```bash
python app.py
```

**Programmatic / single FOV:**
```python
from pipeline import SmartCBC

cbc = SmartCBC()
result = cbc.analyze(image=my_pil_image, age=32, gender="M")
print(result["report_text"])
```

**Multiple FOVs (auto-aggregated):**
```python
result = cbc.analyze(image=[img1, img2, img3], age=32, gender="M")
```

**Test with sample images:**
`test_pipeline.py` expects a local copy of the [TXL-PBC dataset](https://github.com/lugan113/TXL-PBC_Dataset) (the dataset this pipeline's detector classes — RBC/WBC/Platelet — align with). Clone it and point `test_dir` in `test_pipeline.py` at your local `TXL-PBC/images/test` folder, or upload any of its images directly to the Gradio UI / Space.

## Deploying / relaunching the Hugging Face Space

The Space (`KPrashanth/SmartCBC`) is a separate git remote from this repo, not a CI-linked mirror — changes here don't auto-deploy there. Push to the Space's own git remote (`https://huggingface.co/spaces/KPrashanth/SmartCBC`) to update it, or restart it from the Space page ("Restart this Space") if it's just sleeping (default on the free CPU tier after inactivity).

### ⚠️ Known drift between this repo and the live Space

The deployed Space currently contains three fixes that are **not yet present in this repo**. If you sync this repo's code onto the Space as-is, you will regress it:

| File | This repo | Live Space |
|---|---|---|
| `app.py` | `gr.Gallery(type="pil")` for multi-image upload | `gr.Files(file_count="multiple")` + manual PIL conversion (more reliable across Gradio versions) |
| `requirements.txt` | unpinned `gradio` | pinned `gradio==4.44.1` |
| `utils/classifier.py` | `torch.load(weights_path, map_location=DEVICE)` | `torch.load(weights_path, map_location=DEVICE, weights_only=False)` (required on newer PyTorch, which defaults `weights_only=True` and can't unpickle this checkpoint format) |

Backport these three changes into this repo before treating it as the source of truth for a fresh Space deployment.

## Non-diagnostic disclaimer

This is a research prototype. Output is intended for qualitative screening only and is not a substitute for formal laboratory testing or clinical diagnosis.
