import gc
import torch
import gradio as gr

from vlm_describer import VLMDescriber
from outpainting import Outpainter

def generate_descriptions(image):
    describer = VLMDescriber()
    short_description, long_description = describer.describe_image(image)

    del describer

    gc.collect()
    torch.cuda.empty_cache()

    outpainter = Outpainter()
    outpainted_image = outpainter.outpaint(image, short_description)
    
    return outpainted_image, short_description, long_description

if __name__ == "__main__":
    iface = gr.Interface(
        fn=lambda image: generate_descriptions(image),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(label="Generated Photo"),
            gr.Textbox(label="Short Description"),
            gr.Textbox(label="Long Description")
        ],
        title="Image Description and Generation",
        description="Upload an image to receive descriptions and a generated photo.",
        flagging_options=None, allow_flagging="never"
    )

    iface.launch()