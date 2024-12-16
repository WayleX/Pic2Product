import pandas as pd
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from rag.vlm_pipeline import VLMPipeline
from rag.retriever import Retriever, HuggingFaceEmbeddings
from rag.rag_chain import RAGChain

from outpaint.outpainting import Outpainter

from utils import unload_model

import gc
import json
import torch

def setup_vlm():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels=512 * 512)

    return model, processor

def get_categories(cats_csv_path, n = 10):
    cats_df = pd.read_csv(cats_csv_path)
    sampled = cats_df["category_name"].sample(n).values
    print(sampled)
    return list(sampled)

def generate_descriptions(image):
    model, processor = setup_vlm()
    vlm_pipeline = VLMPipeline(model, processor)

    is_product_image = vlm_pipeline(
        "Is this an image of a product? If yes write 'yes'. If image do not looks like a product, write 'no'.", 
        images=[image], 
        system_prompt="You are a quality assurance specialist.",
        max_image_size=512)[0]
    if "no" in is_product_image.lower():
        return image, "No product found", "", "", "No product found"

    system_prompt = "You are an online seller."
    title = vlm_pipeline("Write item name as product label.", images=[image], system_prompt=system_prompt, max_image_size=512)[0]
    print(title)

    category = vlm_pipeline(f"Write item category. Category should sound something like but not restricted to those categories: {get_categories("./data/amazon_categories.csv")}.", 
        images=[image], system_prompt=system_prompt, max_image_size=512)[0]
    print("Category: ", category)

    outpaint_prompt = vlm_pipeline(
        f"Create a very short backround scene caption for the {title}. Only describe scene. Write in style: an {title} on ...",
        system_prompt="You are a professional photographer.",
        images=[],
        max_image_size=512
    )[0]
    print(outpaint_prompt)

    charactheristics = vlm_pipeline(
        f"Write the main visual characteristics of the {title} on photo. Use bulletpoints.",
        system_prompt="You are an online seller.",
        images=[image],
        max_image_size=512,
        
    )[0]

    retriever = Retriever(HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'), store_pickle="chroma_db/retriever_store.pkl")
    rag_chain = RAGChain(retriever, vlm_pipeline)

    desc_retriever = Retriever(
        HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'), 
        chroma_persist_directory="./chroma_desc_db",
        store_pickle="./chroma_db/retriever_desc_store.pkl"
    )

    desc_examples = desc_retriever(title, limit=2)
    descs = [f"\nEXAMPLE {i}: " + json.loads(desc)["DESCRIPTION"] for i, desc in enumerate(desc_examples)]

    del desc_retriever
    del retriever

    gc.collect()
    torch.cuda.empty_cache()

    descs = " ".join(descs)

    print(descs)

    query = f"This is photos of product: {title}. Describe this product. Use next examples as a reference: {descs}"
    long_description = rag_chain(query, image=image, system_prompt=system_prompt, max_image_size=256, retrieval_limit=2)[0]
    print(long_description)

    del model
    del processor
    del rag_chain
    del vlm_pipeline

    gc.collect()
    torch.cuda.empty_cache()

    outpainter = Outpainter()
    outpainted_image = outpainter.outpaint(image, outpaint_prompt)
    
    return outpainted_image, title, category, charactheristics, long_description

if __name__ == "__main__":
    iface = gr.Interface(
        fn=lambda image: generate_descriptions(image),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(label="Generated Photo"),
            gr.Textbox(label="Title"),
            gr.Textbox(label="Category"),
            gr.Textbox(label="Characteristics"),
            gr.Textbox(label="Description")
        ],
        title="Image Description and Generation",
        description="Upload an image to receive descriptions and a generated photo.",
        flagging_options=None, allow_flagging="never"
    )

    iface.launch()