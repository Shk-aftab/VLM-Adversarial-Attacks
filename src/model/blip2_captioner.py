import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2Captioner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Generation controls
        self.max_length = 30  # interpreted as max_new_tokens for BLIP-2
        self.single_sentence = True  # can be overridden by runner
        # Optional instruction/prompt (may be set by runner)
        self.prompt = None

    @torch.no_grad()
    def caption(self, pil_image, max_length: int | None = None):
        # For BLIP-2, prefer controlling length via max_new_tokens to avoid coupling with input prompt length
        ml = max_length if max_length is not None else self.max_length
        if getattr(self, "prompt", None):
            inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=ml)
        txt = self.processor.decode(out[0], skip_special_tokens=True)
        if self.single_sentence:
            # Trim to first sentence-like segment
            for sep in ['.', '!', '?']:
                if sep in txt:
                    txt = txt.split(sep)[0]
                    break
            txt = txt.strip()
        return txt

    def nll_loss(self, pil_image, text: str) -> torch.Tensor:
        inputs = self.processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss

    def nll_loss_batch(self, pil_images, texts) -> torch.Tensor:
        inputs = self.processor(images=pil_images, text=texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss

    # --- Attack-loop adapters ---
    def prepare_ref_tokens(self, text: str) -> dict:
        ti = self.processor(text=text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in ti.items()}

    def loss_from_pixel_values(self, pixel_values, ref_tokens: dict) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=ref_tokens.get("input_ids"),
            attention_mask=ref_tokens.get("attention_mask"),
            labels=ref_tokens.get("input_ids"),
        )
        return outputs.loss