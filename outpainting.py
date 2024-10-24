import random
import gc

import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

from background_remove import BackgroundRemoval

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

class Outpainter:
    def __init__(self):
        self.controlnets = [
            ControlNetModel.from_pretrained(
                "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
            ),
            ControlNetModel.from_pretrained(
                "diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16
            ),
        ]

        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=self.controlnets, vae=self.vae
        )

        self.refiner_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "OzzyGT/RealVisXL_V4.0_inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            vae=self.vae,
        )

        self.zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        self.background_remover = BackgroundRemoval()

    @staticmethod
    def scale_and_paste(original_image):
        aspect_ratio = original_image.width / original_image.height

        if original_image.width > original_image.height:
            new_width = 1024
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = 1024
            new_width = round(new_height * aspect_ratio)

        resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
        white_background = Image.new("RGBA", (1024, 1024), "white")
        x = (1024 - new_width) // 2
        y = (1024 - new_height) // 2
        white_background.paste(resized_original, (x, y), resized_original)

        return resized_original, white_background
    
    def _main_generate_image(self, prompt, negative_prompt, inpaint_image, zoe_image, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=[inpaint_image, zoe_image],
            guidance_scale=6.5,
            num_inference_steps=25,
            generator=generator,
            controlnet_conditioning_scale=[0.5, 0.8],
            control_guidance_end=[0.9, 0.6],
        ).images[0]

        return image
    
    def _refiner_generate_outpaint(self, prompt, negative_prompt, image, mask, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = self.refiner_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=10.0,
            strength=0.8,
            num_inference_steps=25,
            generator=generator,
        ).images[0]

        return image
    
    def _blurred_mask(self, resized_img: Image, temp_image: Image, x: int, y: int):
        mask = Image.new("L", temp_image.size)
        mask.paste(resized_img.split()[3], (x, y))
        mask = ImageOps.invert(mask)
        final_mask = mask.point(lambda p: p > 128 and 255)
        mask_blurred = self.refiner_pipeline.mask_processor.blur(final_mask, blur_factor=20)
        return mask_blurred
    
    def outpaint(self, image: Image, prompt: str):
        self.background_remover.to_device("cuda")
        original_image = self.background_remover.remove_background(image)
        unload_model(self.background_remover)

        resized_img, white_bg_image = self.scale_and_paste(original_image)

        image_zoe = self.zoe(white_bg_image, detect_resolution=512, image_resolution=1024)
        unload_model(self.zoe)

        self.pipeline.to("cuda")
        temp_image = self._main_generate_image(prompt, "", white_bg_image, image_zoe)
        unload_model(self.pipeline)

        x = (1024 - resized_img.width) // 2
        y = (1024 - resized_img.height) // 2
        temp_image.paste(resized_img, (x, y), resized_img)

        temp_image.save("temp_image.png")

        self.refiner_pipeline.enable_sequential_cpu_offload()
        mask_blurred = self._blurred_mask(resized_img, temp_image, x, y)
        final_image = self._refiner_generate_outpaint("item on a stone near waterfall", "", temp_image, mask_blurred)

        final_image.paste(resized_img, (x, y), resized_img)
        return final_image


if __name__ == "__main__":
    outpainter = Outpainter()
    image = Image.open("2.jpg")
    image = outpainter.outpaint(image, "water bottle on a stone near waterfall")

    image.save("outpaint_image.png")