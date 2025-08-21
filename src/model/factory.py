import importlib

def create_captioner(model_name: str, device: str):
    """
    Return a captioner instance based on model_name.
    - Contains "llava" -> LLaVACaptioner
    - Contains "blip2" -> BLIP2Captioner
    - Else -> BLIPCaptioner
    Lazy-imports to avoid unnecessary heavy deps when unused.
    """
    name_l = (model_name or "").lower()

    if "llava" in name_l:
        mod = importlib.import_module("src.model.llava_captioner")
        return mod.LLaVACaptioner(model_name, device)
    if "blip2" in name_l:
        mod = importlib.import_module("src.model.blip2_captioner")
        return mod.BLIP2Captioner(model_name, device)

    mod = importlib.import_module("src.model.blip_captioner")
    return mod.BLIPCaptioner(model_name, device)

def get_model_family(model_name: str) -> str:
    """Return a short family tag for directory naming: 'llava', 'blip2', or 'blip'."""
    name_l = (model_name or "").lower()
    if "llava" in name_l:
        return "llava"
    if "blip2" in name_l:
        return "blip2"
    return "blip"
