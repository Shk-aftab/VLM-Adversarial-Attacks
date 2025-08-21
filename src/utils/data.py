import json, os, random
from PIL import Image

class CocoValDataset:
    def __init__(self, data_root: str, split: str = "val2017", max_images: int | None = None):
        self.images_dir = os.path.join(data_root, split)
        ann_path = os.path.join(data_root, "annotations", "captions_val2017.json")
        with open(ann_path, "r") as f:
            ann = json.load(f)
        id2file = {img["id"]: img["file_name"] for img in ann["images"]}
        caps = {}
        for a in ann["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"])
        items = []
        for image_id, file_name in id2file.items():
            if image_id in caps:
                items.append({"image_id": image_id, "file_name": file_name, "captions": caps[image_id]})
        random.seed(123); random.shuffle(items)
        if max_images is not None:
            items = items[:max_images]
        self.items = items

    def __len__(self): return len(self.items)

    def __iter__(self):
        for it in self.items:
            path = os.path.join(self.images_dir, it["file_name"])
            img = Image.open(path).convert("RGB")
            yield {**it, "image": img}
