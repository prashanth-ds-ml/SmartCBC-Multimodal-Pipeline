# app.py

import gradio as gr
from PIL import Image

from pipeline import SmartCBC

# -------------------------------------------------
# Initialize pipeline ONCE (cached in Spaces)
# -------------------------------------------------
cbc = SmartCBC()   # loads YOLO + classifier once


# -------------------------------------------------
# Core Gradio wrapper
# -------------------------------------------------
def analyze_images(images, age, gender, output_mode):
    """
    Wrapper for SmartCBC.analyze() or SmartCBC.analyze_batch()
    depending on whether 1 or multiple images are uploaded.
    """
    if images is None or len(images) == 0:
        return "Please upload at least one image.", None

    # Gradio Gallery sometimes returns list of tuples: (PIL.Image, metadata)
    def to_pil_list(img_list):
        pil_list = []
        for item in img_list:
            if isinstance(item, tuple):
                # assume first element is the PIL image
                pil = item[0]
            else:
                pil = item
            pil_list.append(pil)
        return pil_list

    pil_images = to_pil_list(images)

    # If 1 image → single-FOV path
    if len(pil_images) == 1:
        result = cbc.analyze(
            image=pil_images[0],
            age=age,
            gender=gender
        )
    else:
        # Multiple images → multi-FOV aggregation
        result = cbc.analyze(
            image=pil_images,   # list → analyze_batch under the hood
            age=age,
            gender=gender
        )

    if output_mode == "Text Report":
        return result["report_text"], None
    else:
        return None, result


# -------------------------------------------------
# Gradio UI Layout
# -------------------------------------------------
with gr.Blocks(title="SmartCBC - Multimodal Blood Analysis") as demo:

    gr.Markdown("""
    # 🩸 SmartCBC — Multimodal AI Blood Smear Analysis  
    Upload **one or multiple** peripheral smear FOV images and get:  
    - RBC / WBC / Platelet counts  
    - WBC subtype classification  
    - Aggregated multi-FOV differential  
    - Age-specific reference comparisons  
    - Clinical insights (non-diagnostic)  
    """)

    with gr.Row():
        img_in = gr.Gallery(
            label="Upload 1 or Multiple Blood Smear Images (FOVs)",
            columns=3,
            height="auto",
            allow_preview=True,
            type="pil"
        )

        with gr.Column():
            age_in = gr.Number(label="Age (years)", value=30)
            gender_in = gr.Dropdown(
                ["M", "F", ""],
                label="Gender (optional)",
                value=""
            )
            output_mode = gr.Radio(
                ["Text Report", "Structured JSON"],
                value="Text Report",
                label="Output Format"
            )
            btn = gr.Button("Analyze")

    # OUTPUT AREAS
    txt_out = gr.Textbox(
        label="Report (Human Readable)",
        visible=True,
        lines=30,
        interactive=False
    )

    json_out = gr.JSON(
        label="Structured Output (JSON)",
        visible=False
    )

    # Toggle visibility
    def toggle_output(mode):
        return (
            gr.update(visible=(mode == "Text Report")),
            gr.update(visible=(mode == "Structured JSON"))
        )

    output_mode.change(toggle_output, [output_mode], [txt_out, json_out])

    # Button Binding
    btn.click(
        analyze_images,
        inputs=[img_in, age_in, gender_in, output_mode],
        outputs=[txt_out, json_out]
    )


# -------------------------------------------------
# HF Spaces entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    demo.launch()
