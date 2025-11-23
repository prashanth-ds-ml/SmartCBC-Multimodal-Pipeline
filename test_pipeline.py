# test_pipeline.py

from pathlib import Path
from PIL import Image

from pipeline import SmartCBC

def main():
    # 1) Initialize pipeline (loads YOLO + classifier)
    cbc = SmartCBC()

    # 2) Pick a sample FOV image (from your TXL-PBC test set)
    test_dir = Path("/home/enma/Projects/blood_analyzer/TXL-PBC_Dataset/TXL-PBC/images/test")
    image_paths = list(test_dir.glob("*.*"))

    if not image_paths:
        print(f"No images found in {test_dir}")
        return

    sample_path = image_paths[0]
    print(f"Using sample image: {sample_path}")

    img = Image.open(sample_path).convert("RGB")

    # 3) Run analysis
    result = cbc.analyze(
        image=img,
        age=32,        # you can change this
        gender="M",    # or leave None / ""
    )

    # 4) Print key outputs
    print("\n=== Coarse Counts ===")
    print(result.get("coarse_counts"))

    print("\n=== WBC Subtypes ===")
    print(result.get("wbc_subtypes"))

    print("\n=== WBC Percentages ===")
    print(result.get("wbc_percentages"))

    print("\n=== Report Text (first 40 lines) ===")
    report_lines = result.get("report_text", "").splitlines()
    for line in report_lines[:40]:
        print(line)

if __name__ == "__main__":
    main()
