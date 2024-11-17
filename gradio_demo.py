import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from rag.vlm_pipeline import VLMPipeline
from rag.retriever import Retriever, HuggingFaceEmbeddings
from rag.rag_chain import RAGChain

from outpaint.outpainting import Outpainter

from utils import unload_model

import gc
import torch

def setup_vlm():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels=512 * 512)

    return model, processor

def generate_descriptions(image):
    model, processor = setup_vlm()
    vlm_pipeline = VLMPipeline(model, processor)

    system_prompt = "You are an online seller."

    title = vlm_pipeline("Write item name as product label.", images=[image], system_prompt=system_prompt, max_image_size=512)[0]
    print(title)

    outpaint_prompt = vlm_pipeline(
        f"Create a very short backround scene caption for the {title}. Only describe scene. Write in style: an {title} on ...",
        system_prompt="You are a professional photographer.",
        images=[],
        max_image_size=512
    )[0]
    print(outpaint_prompt)

    retriever = Retriever(HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'), store_pickle="chroma_db/retriever_store.pkl")
    rag_chain = RAGChain(retriever, vlm_pipeline)

    query = f"Consider all images as photos of product: {title}. As a seller describe this product as detailed as you can. Use bulletpoints."
    long_description = rag_chain(query, image=image, system_prompt=system_prompt, max_image_size=256, retrieval_limit=2)[0]
    print(long_description)

    del model
    del processor
    del rag_chain
    del vlm_pipeline
    del retriever

    gc.collect()
    torch.cuda.empty_cache()

    outpainter = Outpainter()
    outpainted_image = outpainter.outpaint(image, outpaint_prompt)
    
    return outpainted_image, title, long_description

if __name__ == "__main__":
    iface = gr.Interface(
        fn=lambda image: generate_descriptions(image),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(label="Generated Photo"),
            gr.Textbox(label="Title", ),
            gr.Textbox(label="Description")
        ],
        title="Image Description and Generation",
        description="Upload an image to receive descriptions and a generated photo.",
        flagging_options=None, allow_flagging="never"
    )

    iface.launch()