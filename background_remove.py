import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize


class BackgroundRemoval:
    def __init__(self):
        self.model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)

    def to_device(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)

    def _preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
        return image

    def _postprocess_image(self, result: torch.Tensor, im_size: list)-> np.ndarray:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def remove_background(self, img: Image):
        image = np.array(img.copy())
        orig_im_size = image.shape[0:2]
        image = self._preprocess_image(image, orig_im_size).to(self.device)
        
        result=self.model(image)

        result_image = self._postprocess_image(result[0][0], orig_im_size)

        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))

        no_bg_image.paste(img, mask=pil_im)
        return no_bg_image
    

if __name__ == "__main__":
    bg_remover = BackgroundRemoval()
    image = Image.open("2.jpg")
    image = bg_remover.remove_background(image)

    image.save("no_bg_image.png")
