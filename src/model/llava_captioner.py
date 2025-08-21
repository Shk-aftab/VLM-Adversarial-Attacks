import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class LLaVACaptioner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Generation controls (can be overridden by runner)
        self.max_length = 30
        self.single_sentence = True
        self.prompt = "Describe the image in one concise sentence."

    @torch.no_grad()
    def caption(self, pil_image, prompt: str | None = None, max_length: int | None = None):
        pr = prompt if prompt is not None else self.prompt
        ml = max_length if max_length is not None else self.max_length
        inputs = self.processor(text=pr, images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_length=ml)
        txt = self.processor.decode(out[0], skip_special_tokens=True)
        if self.single_sentence:
            for sep in ['.', '!', '?']:
                if sep in txt:
                    txt = txt.split(sep)[0]
                    break
            txt = txt.strip()
        return txt

    def nll_loss(self, pil_image, text: str, prompt: str | None = None) -> torch.Tensor:
        pr = prompt if prompt is not None else self.prompt
        inputs = self.processor(text=pr, images=pil_image, return_tensors="pt").to(self.device)
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def nll_loss_batch(self, pil_images, texts, prompt: str | None = None) -> torch.Tensor:
        pr = prompt if prompt is not None else self.prompt
        inputs = self.processor(text=[pr]*len(pil_images), images=pil_images, return_tensors="pt", padding=True).to(self.device)
        labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids.to(self.device)
        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    # --- Attack-loop adapters ---
    def prepare_ref_tokens(self, text: str) -> dict:
        """Tokenize reference text only (prompt-free)."""
        toks = self.processor.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in toks.items()}

    def loss_from_pixel_values(self, pixel_values: torch.Tensor, ref_tokens: dict, prompt: str | None = None) -> torch.Tensor:
        """Compute loss given pixel_values and tokenized reference.
        Optionally accepts a prompt, but by default uses image-only conditioning.
        """
        if prompt:
            inputs = self.processor(text=prompt, images=None, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {}
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=ref_tokens.get("input_ids"),
        )
        return outputs.loss