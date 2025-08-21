# scripts/side_by_side.py
import argparse, json, os, random
from PIL import Image, ImageDraw, ImageFont

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def wrap_text(draw, text, max_w, font):
    if not text: return ""
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w) if cur else w
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return "\n".join(lines)

def scale_to_width(im: Image.Image, width: int) -> Image.Image:
    w, h = im.size
    if w == 0: return im
    s = width / w
    return im.resize((int(w*s), int(h*s)), Image.BICUBIC)

def pad_to_size(im: Image.Image, target_w: int, target_h: int, color="white") -> Image.Image:
    canvas = Image.new("RGB", (target_w, target_h), color)
    x = (target_w - im.width) // 2
    y = (target_h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas

def build_column(img: Image.Image, caption: str, width: int, pad: int, font) -> Image.Image:
    draw_tmp = ImageDraw.Draw(img)
    txt = wrap_text(draw_tmp, caption, width - 2*pad, font)
    bbox = draw_tmp.multiline_textbbox((0,0), txt, font=font, spacing=2) if txt else (0,0,0,0)
    text_h = (bbox[3]-bbox[1]) if txt else 0
    col = Image.new("RGB", (width, img.height + text_h + 3*pad), "white")
    col.paste(img, (0,0))
    if txt:
        draw = ImageDraw.Draw(col)
        draw.multiline_text((pad, img.height + pad), txt, fill=(0,0,0), font=font, spacing=2, align="left")
    return col

def make_panel(orig_png, adv_png, base_cap, adv_cap, img_width=480, pad=8, include_ref=None):
    # load & equalize image sizes
    left  = Image.open(orig_png).convert("RGB")
    right = Image.open(adv_png).convert("RGB")
    left  = scale_to_width(left,  img_width)
    right = scale_to_width(right, img_width)
    H = max(left.height, right.height)
    left  = pad_to_size(left,  img_width, H)
    right = pad_to_size(right, img_width, H)

    font = ImageFont.load_default()

    # ONLY baseline caption on left, ONLY adversarial on right
    left_caption  = f"BASELINE:\n{base_cap or ''}"
    right_caption = f"ADVERSARIAL:\n{adv_cap or ''}"

    col_l = build_column(left,  left_caption,  img_width, pad, font)
    col_r = build_column(right, right_caption, img_width, pad, font)

    # stitch side-by-side
    H2 = max(col_l.height, col_r.height)
    W2 = col_l.width + col_r.width + pad
    canvas = Image.new("RGB", (W2, H2), "white")
    canvas.paste(col_l, (0,0))
    canvas.paste(col_r, (col_l.width + pad, 0))

    # optional single REF footer across both columns
    if include_ref:
        draw = ImageDraw.Draw(canvas)
        ref_txt = f"REF: {include_ref}"
        bbox = draw.multiline_textbbox((0,0), ref_txt, font=font)
        ref_h = bbox[3]-bbox[1]
        out = Image.new("RGB", (W2, H2 + ref_h + 3*pad), "white")
        out.paste(canvas, (0,0))
        draw = ImageDraw.Draw(out)
        draw.text((pad, H2 + pad), ref_txt, fill=(0,0,0), font=font)
        return out

    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adversarial", required=True)
    ap.add_argument("--saved_dir", required=True, help="Folder with *_orig.png/*_adv.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--img_width", type=int, default=480)
    ap.add_argument("--include_ref", action="store_true", help="Add a single REF footer line")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base = load_jsonl(args.baseline)
    adv  = load_jsonl(args.adversarial)
    bmap = {b["file_name"]: b for b in base}

    pairs = []
    for a in adv:
        b = bmap.get(a["file_name"])
        if not b: continue
        stem = os.path.splitext(os.path.basename(a["file_name"]))[0]
        o_png = os.path.join(args.saved_dir, f"{stem}_orig.png")
        d_png = os.path.join(args.saved_dir, f"{stem}_adv.png")
        if os.path.exists(o_png) and os.path.exists(d_png):
            ref = (b.get("references") or [""])[0]
            pairs.append((b.get("hypothesis"), a.get("hypothesis_adv"), ref, o_png, d_png))

    if not pairs:
        print("No aligned pairs with saved PNGs found in:", args.saved_dir)
        return

    random.seed(0)
    random.shuffle(pairs)
    pairs = pairs[:args.num]

    for i, (base_cap, adv_cap, ref_cap, o_png, d_png) in enumerate(pairs, 1):
        panel = make_panel(
            o_png, d_png,
            base_cap, adv_cap,
            img_width=args.img_width,
            include_ref=(ref_cap if args.include_ref else None)
        )
        panel.save(os.path.join(args.out_dir, f"pair_{i:03d}.png"))

    print(f"Wrote {len}(pairs) panels to {args.out_dir}")

if __name__ == "__main__":
    main()
