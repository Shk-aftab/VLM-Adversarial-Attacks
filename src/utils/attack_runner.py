import torch
import numpy as np
from typing import Callable
from PIL import Image

def pixel_values_from_image(processor, pil_image, device):
    return processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

def to_pil_from_pixel_values(processor, pixel_values: torch.Tensor) -> Image.Image:
    """
    Convert BLIP-normalized pixel_values (B,C,H,W) back to a PIL image.
    Works across transformers versions where BlipImageProcessor lacks `postprocess`.
    """
    x = pixel_values.detach().cpu()[0]  # (C,H,W), float
    mean = torch.tensor(processor.image_processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(processor.image_processor.image_std).view(-1, 1, 1)
    x = x * std + mean                     # de-normalize to [0,1] range (approx)
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).round().byte()         # uint8
    x = x.permute(1, 2, 0).numpy()         # (H,W,C)
    return Image.fromarray(x)

def make_loss_fn(model, processor, text: str) -> Callable[[torch.Tensor], torch.Tensor]:
    text_inputs = processor(text=text, return_tensors="pt")
    def loss_fn(pixel_values: torch.Tensor) -> torch.Tensor:
        ti = {k: v.to(pixel_values.device) for k, v in text_inputs.items()}
        outputs = model(pixel_values=pixel_values,
                        input_ids=ti["input_ids"],
                        attention_mask=ti["attention_mask"],
                        labels=ti["input_ids"])
        return outputs.loss
    return loss_fn
