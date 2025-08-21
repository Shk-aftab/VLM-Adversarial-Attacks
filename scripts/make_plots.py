# scripts/make_plots.py
import argparse, os, json, re, math, random
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

from eval.metrics import compute_bleu, compute_rougeL, compute_clipscore

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def align_runs(baseline_rows, adv_rows):
    """Return lists aligned by file_name so metrics compare the same images."""
    base_by_file = {r["file_name"]: r for r in baseline_rows}
    refs, hyps_base, hyps_adv, img_files = [], [], [], []
    for a in adv_rows:
        b = base_by_file.get(a["file_name"])
        if not b:
            continue
        refs.append(b["references"])
        hyps_base.append(b["hypothesis"])
        hyps_adv.append(a["hypothesis_adv"])
        img_files.append(a["file_name"])
    return refs, hyps_base, hyps_adv, img_files

def parse_eps_from_name(path):
    # Try to infer epsilon from filename like "..._eps4_255..." or "..._eps8_255..."
    m = re.search(r"eps(\d+)_?(\d+)?", os.path.basename(path))
    if m:
        num = float(m.group(1))
        den = float(m.group(2)) if m.group(2) else 255.0
        return num / den
    return None

def compute_run_metrics(baseline_path, adv_path, images_root, device="cpu"):
    base = load_jsonl(baseline_path)
    adv = load_jsonl(adv_path)
    refs, hyps_b, hyps_a, img_files = align_runs(base, adv)

    img_paths = [os.path.join(images_root, f) for f in img_files]

    bleu_b  = compute_bleu(refs, hyps_b)
    bleu_a  = compute_bleu(refs, hyps_a)
    rouge_b = compute_rougeL(refs, hyps_b)
    rouge_a = compute_rougeL(refs, hyps_a)
    try:
        clip_b = compute_clipscore(hyps_b, img_paths, device=device)
        clip_a = compute_clipscore(hyps_a, img_paths, device=device)
    except Exception as e:
        print(f"[WARN] CLIPScore failed ({e}); setting NaN.")
        clip_b = clip_a = float("nan")

    return {
        "aligned": len(hyps_a),
        "BLEU_base": bleu_b,  "BLEU_adv": bleu_a,  "BLEU_delta": bleu_a - bleu_b,
        "ROUGE_base": rouge_b,"ROUGE_adv": rouge_a,"ROUGE_delta": rouge_a - rouge_b,
        "CLIP_base": clip_b,  "CLIP_adv": clip_a,  "CLIP_delta": clip_a - clip_b,
    }

def barplot_metric(df, metric, out_png, title=None):
    # metric in {"BLEU","ROUGE","CLIP"}
    labels = df["run"].tolist()
    base_vals = df[f"{metric}_base"].tolist()
    adv_vals  = df[f"{metric}_adv"].tolist()

    x = range(len(labels))
    w = 0.4
    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], base_vals, width=w, label="Baseline")
    plt.bar([i + w/2 for i in x], adv_vals,  width=w, label="Adversarial")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def lineplot_eps(df, metric, out_png, title=None):
    # Only rows with epsilon parsed
    dd = df.dropna(subset=["epsilon"]).sort_values("epsilon")
    if dd.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(dd["epsilon"], dd[f"{metric}_adv"], marker="o", label=f"{metric} (adv)")
    plt.plot(dd["epsilon"], dd[f"{metric}_base"], marker="o", label=f"{metric} (baseline)")
    plt.xlabel("epsilon (L∞)")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def make_qual_panels(baseline_path, adv_paths, images_root, out_dir, k=12, seed=0):
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(out_dir, exist_ok=True)
    base = load_jsonl(baseline_path)
    base_by_file = {r["file_name"]: r for r in base}

    # Use the first adv run for selection
    adv = load_jsonl(adv_paths[0])
    random.seed(seed)
    random.shuffle(adv)
    adv = adv[:k]

    def draw_row(img_path, base_cap, adv_caps, width=700, text_pad=120):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(width / w, 1.0)
        img = img.resize((int(w * scale), int(h * scale)))
        # Canvas height: image + text area (baseline + each attack line)
        lines = 1 + len(adv_caps)
        canvas = Image.new("RGB", (img.width, img.height + text_pad + 20 * lines), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        y = img.height + 8
        draw.text((8, y), f"Baseline: {base_cap}", fill=(0, 0, 0)); y += 20
        for tag, cap in adv_caps:
            draw.text((8, y), f"{tag}: {cap}", fill=(0, 0, 0)); y += 20
        return canvas

    # Build one big panel
    rows = []
    for r in adv:
        file_name = r["file_name"]
        base_cap = base_by_file[file_name]["hypothesis"] if file_name in base_by_file else ""
        img_path = os.path.join(images_root, file_name)
        adv_caps = []
        for p in adv_paths:
            tag = os.path.splitext(os.path.basename(p))[0]
            rr = [x for x in load_jsonl(p) if x["file_name"] == file_name]
            if rr:
                adv_caps.append((tag, rr[0]["hypothesis_adv"]))
        rows.append(draw_row(img_path, base_cap, adv_caps))

    # Tile rows vertically
    total_h = sum(im.height for im in rows)
    max_w = max(im.width for im in rows)
    panel = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in rows:
        panel.paste(im, (0, y))
        y += im.height
    out_path = os.path.join(out_dir, "qualitative_panel.png")
    panel.save(out_path)
    print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline jsonl from scripts/caption_images.py")
    ap.add_argument("--adversarial", nargs="+", required=True, help="one or more adversarial jsonl files")
    ap.add_argument("--images_root", required=True, help="folder with images (e.g., data/coco2017/val2017)")
    ap.add_argument("--out_dir", default="eval/plots")
    ap.add_argument("--device", default="cpu", help="device for CLIPScore ('cuda' if available)")
    ap.add_argument("--make_qual", action="store_true", help="also save a qualitative panel image")
    ap.add_argument("--qual_k", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute metrics per run
    rows = []
    for adv in args.adversarial:
        m = compute_run_metrics(args.baseline, adv, args.images_root, device=args.device)
        rows.append({
            "run": os.path.splitext(os.path.basename(adv))[0],
            "epsilon": parse_eps_from_name(adv),
            **m
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Bar charts
    for metric in ["BLEU", "ROUGE", "CLIP"]:
        barplot_metric(df, metric, os.path.join(args.out_dir, f"bar_{metric}.png"),
                       title=f"{metric}: Baseline vs Adversarial")

    # Epsilon sweep charts (if epsilon detected)
    for metric in ["BLEU", "ROUGE", "CLIP"]:
        lineplot_eps(df, metric, os.path.join(args.out_dir, f"sweep_{metric}.png"),
                     title=f"{metric} vs epsilon")

    # Qualitative panel (baseline + each attack captions stacked, for K examples)
    if args.make_qual:
        make_qual_panels(args.baseline, args.adversarial, args.images_root, args.out_dir, k=args.qual_k)

if __name__ == "__main__":
    main()
