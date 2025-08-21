import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPCaptioner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, pil_image, max_length: int = 30):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_length=max_length)
        txt = self.processor.decode(out[0], skip_special_tokens=True)
        return txt
    

    def nll_loss(self, pil_image, text: str) -> torch.Tensor:
        """Compute negative log-likelihood of the given caption (as labels)."""
        inputs = self.processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss
    
    def nll_loss_batch(self, pil_images, texts) -> torch.Tensor:
        """Compute negative log-likelihood for multiple images/texts in batch."""
        inputs = self.processor(images=pil_images, text=texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss
