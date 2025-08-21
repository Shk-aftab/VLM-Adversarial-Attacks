import argparse, json, os, random
from PIL import Image, ImageDraw

def load_jsonl(path):
    recs = []
    with open(path, "r") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def draw_panel(img_path, caption, width=640):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = width / w
    img = img.resize((int(w*scale), int(h*scale)))
    pad = 100
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0,0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, img.height + 10), caption, fill=(0,0,0))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = load_jsonl(args.results)
    random.seed(0); random.shuffle(recs)
    recs = recs[:args.num]

    for i, r in enumerate(recs, 1):
        img_path = os.path.join(args.images_root, r["file_name"])
        cap = r.get("hypothesis_adv", r.get("hypothesis", ""))
        panel = draw_panel(img_path, cap)
        panel.save(os.path.join(args.out_dir, f"example_{i:03d}.png"))

if __name__ == "__main__":
    main()
