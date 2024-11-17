import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPSimularity:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def compute_similarity(self, image1: Image, image2: Image):
        inputs = self.processor(images=[image1, image2], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_image_features(**inputs)

        similarity = torch.nn.functional.cosine_similarity(outputs[0], outputs[1], dim=0)
        return similarity.item()
        
if __name__ == "__main__":
    clip = CLIPSimularity()
    similarity = clip.compute_similarity(Image.open("images/2.jpg"), Image.open("images/3.jpg"))
    print(similarity)