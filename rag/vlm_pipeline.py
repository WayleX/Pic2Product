from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration

from transformers import Pipeline
from PIL import Image


class VLMPipeline(Pipeline):
    def __init__(self, model, processor, *args, **kwargs):
        super().__init__(model, None, *args, **kwargs)
        self.my_processor = processor

    @staticmethod
    def __resize_ratio_preserve(image: Image, max_size: int = 512):
        if max(image.size) <= max_size:
            return image

        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
        return resized_image
    
    def preprocess(self, query, images, max_image_size=512, system_prompt:str = ""):
        prompt_template = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "image"} for _ in range(len(images))
                ] + [
                    {"type": "text", "text": query},
                ],
            }
        ]

        if system_prompt:
            sysmessage = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }

            prompt_template.insert(0, sysmessage)

        images = [VLMPipeline.__resize_ratio_preserve(image, max_image_size) for image in images]
        print([image.size for image in images])

        if not images:
            images = None

        text_prompt = self.my_processor.apply_chat_template(prompt_template, add_generation_prompt=True)
        inputs = self.my_processor(
            text=[text_prompt], images=images, padding=True, return_tensors="pt"
        )

        inputs = inputs.to("cuda")
        return inputs
    
    def _forward(self, inputs):
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        
        return generated_ids
    
    def postprocess(self, outputs):
        output_text = self.my_processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text

    def _sanitize_parameters(self, **kwargs):
        preprocessor_kwargs = {}
        forward_kwargs = {}

        if "images" in kwargs:
            preprocessor_kwargs["images"] = kwargs["images"]

        if "max_image_size" in kwargs:
            preprocessor_kwargs["max_image_size"] = kwargs["max_image_size"]

        if "max_new_tokens" in kwargs:
            forward_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

        if "system_prompt" in kwargs:
            preprocessor_kwargs["system_prompt"] = kwargs["system_prompt"]

        return preprocessor_kwargs, forward_kwargs, {}
    

if __name__ == "__main__":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels=512 * 512)

    image = Image.open("images/2.jpg")
    image2 = Image.open("images/3.jpg")

    pipeline = VLMPipeline(model, processor, max_image_size=512)
    
    query = "Given the images, describe what are distinct features of the two products on the images."
    description = pipeline(query, images=[image, image2])
    print(description)