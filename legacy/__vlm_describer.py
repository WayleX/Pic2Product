from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor


from utils import unload_model
from outpaint.background_remove import BackgroundRemoval

class VLMDescriber:
    image_title_conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Write item name as product label."},
            ],
        }
    ]

    image_desc_conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Describe this image as AliExpress product."},
            ],
        }
    ]

    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels=512*512)

    def _preprocess_image(self, image: Image, conversation: Dict):
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        return inputs
    
    def _generate_image_description(self, inputs):
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text
    
    def describe_image(self, image: Image):
        # self.model.to("
        # self.processor.to("cuda")

        image = image.resize((512, 512))
        inputs_1 = self._preprocess_image(image, self.image_title_conversation)
        inputs_2 = self._preprocess_image(image, self.image_desc_conversation)

        title = self._generate_image_description(inputs_1)
        description = self._generate_image_description(inputs_2)

        return title, description


if __name__ == "__main__":
    bg_remover = BackgroundRemoval()
    bg_remover.to_device("cuda")
    image = Image.open("2.jpg")
    image = bg_remover.remove_background(image)

    describer = VLMDescriber()
    description = describer.describe_image(image)
    print(description)

    image.save("no_bg_image.png")