import requests
from io import BytesIO
from PIL import Image

from .retriever import Retriever
from .vlm_pipeline import VLMPipeline
from .clip_compare import CLIPSimularity

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from utils import unload_model

import gc
import torch


class RAGChain:
    @staticmethod
    def url_to_pil(url):
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content))
        return pil_image

    def __init__(self, retriever, vlm_pipeline):
        self.retriever = retriever
        self.vlm_pipeline = vlm_pipeline

    def __call__(self, query, image, max_image_size=512, retrieval_limit=6):
        docs = self.retriever(query)[:retrieval_limit]
        doc_images = [RAGChain.url_to_pil(doc) for doc in docs]

        clip = CLIPSimularity()
        doc_images = [doc_img for doc_img in doc_images if clip.compute_similarity(doc_img, image) > 0.5]
        
        del clip
        gc.collect()
        torch.cuda.empty_cache()

        doc_images.append(image)

        return self.vlm_pipeline(query, images=doc_images, max_image_size=max_image_size)
    

if __name__ == "__main__":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels=512 * 512)
    vlm_pipeline = VLMPipeline(model, processor)

    retriever = Retriever(
        HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={"device": "cpu"}),
        chroma_persist_directory="./chroma_db", 
        store_pickle="chroma_db/retriever_store.pkl"
    )

    rag_chain = RAGChain(retriever, vlm_pipeline)

    query = "Given the images of water bottles(last is the original photo), generate a description of the the product as on ALIEXPRESS."
    image = Image.open("images/2.jpg")
    
    output_text = rag_chain(query, image, max_image_size=256, retrieval_limit=1)

    print(output_text)
        