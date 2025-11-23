# utils/classifier.py

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from utils.analysis import CLASS_NAMES

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
def load_wbc_classifier(weights_path: str | Path):
    """
    Load your trained ResNet50 classifier.
    Expected checkpoint format:
        {"model_state_dict": ..., ...}
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights_path}")

    # Base model (ImageNet pre-trained)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace FC layer with your 8-class head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )

    # Load checkpoint
    ckpt = torch.load(weights_path, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


# -------------------------------------------------
# SINGLE-CROP CLASSIFICATION
# -------------------------------------------------
def classify_wbc_crop(
    model: nn.Module,
    pil_img: Image.Image,
) -> str:
    """
    Run classification on a single crop and return predicted class name.
    """
    x = clf_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    return CLASS_NAMES[pred_idx]
