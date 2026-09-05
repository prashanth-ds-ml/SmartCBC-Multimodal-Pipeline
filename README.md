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

The deployed Space currently contains fixes that are **not yet present in this repo**. If you sync this repo's code onto the Space as-is, you will regress it:

| File | This repo | Live Space |
|---|---|---|
| `app.py` | `gr.Gallery(type="pil")` for multi-image upload | `gr.Files(file_count="multiple")` + manual PIL conversion (more reliable across Gradio versions) |
| `requirements.txt` | unpinned `gradio` | pinned `gradio==6.26.0`, no other manual pins (see incident notes below) |
| `README.md` (Space config) | n/a | `sdk_version: "6.26.0"`, `python_version: "3.11"` |
| `utils/classifier.py` | `torch.load(weights_path, map_location=DEVICE)` | `torch.load(weights_path, map_location=DEVICE, weights_only=False)` (required on newer PyTorch, which defaults `weights_only=True` and can't unpickle this checkpoint format) |

Backport these changes into this repo before treating it as the source of truth for a fresh Space deployment.

### 📋 2026-09-04/05 outage incident (Space)

A docs-only `README.md` push to the Space triggered a full Docker cache-miss rebuild, which surfaced a chain of *latent* dependency conflicts that had been masked by cached layers for a long time:

1. `sdk_version: "4.0.0"` conflicted with `requirements.txt`'s `gradio==4.44.1` pin (HF's build injects `gradio[oauth]==<sdk_version>` into the same pip install line as `requirements.txt` — mismatched versions fail to resolve) → `BUILD_ERROR`.
2. Fixed the version match, but the base image resolved Python 3.13, which dropped the stdlib `audioop` module that Gradio's `pydub` import chain needs (even with zero audio components in this app) → `RUNTIME_ERROR`. Fixed with `python_version: "3.11"`.
3. Unpinned `huggingface_hub` resolved to a release that removed the `HfFolder` class, which `gradio==4.44.1`'s `oauth.py` imports → `RUNTIME_ERROR`.
4. Unpinned `pydantic` resolved to `>=2.10`, which changed how `additionalProperties` is emitted in JSON schemas (bare `bool` instead of `dict`), crashing gradio 4.44.1's bundled `gradio_client` schema parser → `RUNTIME_ERROR`.
5. A further Jinja2/`TemplateResponse` "unhashable type: dict" crash on `demo.launch()` appeared next, whose exact trigger was never fully isolated.

Rather than keep patching individual transitive-dependency pins against an aging `gradio==4.44.1` release fighting today's PyPI-latest packages, the fix was to **upgrade to a current, actively-maintained Gradio release (6.26.0)** and let pip resolve its own compatible dependency set with no manual pins — verified end-to-end locally (real `app.py`, real model weights, a real sample image through the full pipeline via `gradio_client`) before pushing.

**Lesson for next time:** reproduce in a local throwaway venv first — install the exact same `requirements.txt`, run the real `app.py`, curl `http://127.0.0.1:<port>/`, then drive a real prediction through `gradio_client.Client(...).predict(..., api_name=...)` — before pushing to the Space and waiting on a remote rebuild. Also: **any** push to the Space repo, even a docs-only one, can force a fresh uncached build and expose previously-hidden dependency conflicts — watch the Space's build status after every push, not just after code changes.

## Non-diagnostic disclaimer

This is a research prototype. Output is intended for qualitative screening only and is not a substitute for formal laboratory testing or clinical diagnosis.
