import argparse, json, os, sys
import wandb

# Add the workspace root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import compute_bleu, compute_rougeL, compute_clipscore

def load_jsonl(path):
    recs = []
    with open(path, "r") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adversarial", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="Log results to wandb")
    ap.add_argument("--wandb_run_name", help="Name for wandb run")
    args = ap.parse_args()
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="adversarial-blip-attacks",
            name=args.wandb_run_name or "evaluation",
            tags=["evaluation", "metrics"]
        )

    base = load_jsonl(args.baseline)
    adv = load_jsonl(args.adversarial)
    ref_map = {b["file_name"]: b for b in base}

    refs, hyps_base, hyps_adv, img_paths = [], [], [], []
    for a in adv:
        b = ref_map.get(a["file_name"])
        if not b: continue
        refs.append(b["references"])
        hyps_base.append(b["hypothesis"])
        hyps_adv.append(a["hypothesis_adv"])
        img_paths.append(os.path.join(args.images_root, a["file_name"]))

    print(f"Aligned examples: {len(hyps_adv)}")
    bleu_base = compute_bleu(refs, hyps_base)
    bleu_adv = compute_bleu(refs, hyps_adv)
    rouge_base = compute_rougeL(refs, hyps_base)
    rouge_adv = compute_rougeL(refs, hyps_adv)
    try:
        clip_base = compute_clipscore(hyps_base, img_paths)
        clip_adv = compute_clipscore(hyps_adv, img_paths)
    except Exception as e:
        print(f"CLIPScore failed: {e}")
        clip_base = clip_adv = float("nan")

    # Calculate deltas
    bleu_delta = bleu_adv - bleu_base
    rouge_delta = rouge_adv - rouge_base
    clip_delta = clip_adv - clip_base
    
    print("\n== Metrics ==")
    print(f"BLEU  - baseline: {bleu_base:.4f}  | adversarial: {bleu_adv:.4f} | delta: {bleu_delta:.4f}")
    print(f"ROUGE - baseline: {rouge_base:.4f} | adversarial: {rouge_adv:.4f} | delta: {rouge_delta:.4f}")
    print(f"CLIP  - baseline: {clip_base:.4f}  | adversarial: {clip_adv:.4f} | delta: {clip_delta:.4f}")
    
    # Log to wandb if enabled
    if args.wandb:
        metrics = {
            "bleu_baseline": bleu_base,
            "bleu_adversarial": bleu_adv,
            "bleu_delta": bleu_delta,
            "rouge_baseline": rouge_base,
            "rouge_adversarial": rouge_adv,
            "rouge_delta": rouge_delta,
            "clip_baseline": clip_base,
            "clip_adversarial": clip_adv,
            "clip_delta": clip_delta,
            "aligned_examples": len(hyps_adv)
        }
        wandb.log(metrics)
        
        # Log sample comparisons
        sample_comparisons = []
        for i in range(min(10, len(hyps_base))):
            sample_comparisons.append([
                img_paths[i].split('/')[-1],  # filename
                hyps_base[i],
                hyps_adv[i],
                str(refs[i])
            ])
        
        wandb.log({
            "caption_comparisons": wandb.Table(
                columns=["image", "baseline_caption", "adversarial_caption", "references"],
                data=sample_comparisons
            )
        })
        
        wandb.finish()

    if args.lpips:
        print("LPIPS: For a proper LPIPS, run compute_lpips_psnr_ssim.py on the saved images.")
    
    return {
        "bleu_baseline": bleu_base,
        "bleu_adversarial": bleu_adv,
        "bleu_delta": bleu_delta,
        "rouge_baseline": rouge_base,
        "rouge_adversarial": rouge_adv,
        "rouge_delta": rouge_delta,
        "clip_baseline": clip_base,
        "clip_adversarial": clip_adv,
        "clip_delta": clip_delta
    }

if __name__ == "__main__":
    main()
