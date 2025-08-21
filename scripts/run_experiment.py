#!/usr/bin/env python3
"""
Unified experiment runner that uses config files and integrates wandb tracking.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import wandb
import nltk
from PIL import Image
import glob
import numpy as np
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import lpips

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_experiment_config, parse_float_maybe_frac
from src.model.factory import create_captioner
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.veattack import veattack
from src.utils.attack_runner import to_pil_from_pixel_values, pixel_values_from_image
from eval.metrics import compute_bleu, compute_rougeL, compute_clipscore
from src.utils.data import CocoValDataset

# captioner factory now provided by src.model.factory.create_captioner

def run_baseline(config: dict, captioner, dataset: CocoValDataset):
    """Run baseline captioning without attacks."""
    # Store baseline at eval/<base_name> so it can be reused across runs
    base_name = os.path.basename(config["output_file"])
    output_file = os.path.join("eval", base_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # If baseline already exists, skip recomputation
    if os.path.exists(output_file):
        print(f"Baseline already exists at {output_file}; skipping generation.")
        return output_file
    
    captions = []
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Baseline captioning"):
            hyp = captioner.caption(item["image"])
            rec = {
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "hypothesis": hyp,
                "references": item["captions"]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            captions.append(hyp)
    
    # Log sample captions to wandb
    wandb.log({
        "sample_captions": wandb.Table(
            columns=["image_id", "caption", "references"],
            data=[[item["image_id"], hyp, str(item["captions"])] 
                  for item, hyp in zip(list(dataset)[:10], captions[:10])]
        )
    })
    
    return output_file


def run_attack(config: dict, captioner, dataset: CocoValDataset):
    """Run adversarial attack experiment (fast path).
    - Runs on ALL images
    - Writes JSONL for ALL
    - Logs up to N examples (default 50) to W&B
    - No image files saved to disk
    """
    import os, json, math
    from tqdm import tqdm
    import torch
    import wandb

    # ---- Core params ----
    attack_type = config["attack"]
    eps = parse_float_maybe_frac(config["epsilon"])
    output_file = config["output_file"]
    alpha = parse_float_maybe_frac(config.get("alpha", "1/255"))
    steps = int(config.get("steps", 10))
    ve_mode = config.get("ve_mode", "max")
    ve_momentum = float(config.get("ve_momentum", 0.9))
    wandb_display_limit = int(
        config.get("wandb_display_limit")
        or (config.get("save_images", {}) or {}).get("max_count", 10)
        or 10
    )

    # ---- I/O setup ----
    # Route output under eval/<model>/<run_mode>/
    model_name = config["model"]["name"] if config.get("model") else "model"
    model_safe = model_name.replace('/', '__')
    run_mode = config.get("run_mode") or ("high_impact" if "high" in (config.get("wandb", {}).get("project", "").lower()) else "standard")
    base_name = os.path.basename(output_file)
    output_file = os.path.join("eval", model_safe, run_mode, base_name)
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    # Optional saving of image pairs for image-quality metrics
    save_cfg = config.get("save_images", {}) or {}
    save_images_enabled = bool(save_cfg.get("enabled", False))
    max_images_to_save = save_cfg.get("max_count")
    save_dir = config.get("save_dir")
    save_root = None
    if save_images_enabled:
        # Use provided save_dir if present; otherwise derive from output_file
        if not save_dir:
            stem = os.path.splitext(os.path.basename(output_file))[0]
            # Place under runs/<model>/<run_mode>/<stem>
            save_dir = os.path.join("runs", model_safe, run_mode, stem)
            config["save_dir"] = save_dir
        else:
            # If provided, normalize under runs/<model>/<run_mode>/ using its basename
            base = os.path.basename(save_dir.rstrip(os.sep))
            save_dir = os.path.join("runs", model_safe, run_mode, base)
            config["save_dir"] = save_dir
        # Save directly under the provided/derived run directory (flat layout)
        save_root = save_dir
        os.makedirs(save_root, exist_ok=True)

    # ---- Model setup: eval, freeze, amp policies ----
    captioner.model.eval()
    for p in captioner.model.parameters():
        p.requires_grad_(False)

    # When generating captions or computing clean/adv losses for LOGGING,
    # we can safely use autocast. During ATTACK updates, keep full precision.
    use_cuda = torch.cuda.is_available()
    amp_dtype = torch.float16 if use_cuda else None

    # ---- W&B: log attack hyperparams ----
    attack_params = {
        "attack_type": attack_type,
        "epsilon": eps,
        "epsilon_str": config["epsilon"],
    }
    if attack_type == "pgd":
        attack_params.update({
            "alpha": alpha,
            "alpha_str": config.get("alpha", "1/255"),
            "steps": steps,
        })
    elif attack_type == "veattack":
        attack_params.update({
            "alpha": alpha,
            "alpha_str": config.get("alpha", "1/255"),
            "steps": steps,
            "ve_mode": ve_mode,
            "ve_momentum": ve_momentum,
        })
    wandb.log(attack_params)

    # ---- Caches ----
    # Cache tokenized refs to avoid tokenizer calls every PGD step
    ref_token_cache = {}

    def get_tokenized_ref(ref: str):
        ti = ref_token_cache.get(ref)
        if ti is None:
            # Use model-agnostic tokenizer preparation
            ti = captioner.prepare_ref_tokens(ref)
            ref_token_cache[ref] = ti
        return ti

    attack_success_count = 0
    total_loss_increase = 0.0
    caption_comparisons = []
    images_saved_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Running {attack_type.upper()} attack")):
            ref = item["captions"][0]
            ti = get_tokenized_ref(ref)

            # Prepare input tensor once per image
            pv = pixel_values_from_image(
                captioner.processor, item["image"], captioner.device
            )

            # ---- Caption (clean) - fast path, no grads, amp ok ----
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        orig_caption = captioner.caption(item["image"])
                else:
                    orig_caption = captioner.caption(item["image"])

            # ---- Define loss fn (uses cached tokens) ----
            def loss_fn(x):
                # Use model-agnostic loss function
                return captioner.loss_from_pixel_values(x, ti)

            # ---- Clean loss (for logging only) ----
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        orig_loss = float(loss_fn(pv).item())
                else:
                    orig_loss = float(loss_fn(pv).item())

            # ---- Run ATTACK (full precision; grads enabled only here) ----
            # Make sure pv requires grad only inside attack
            if attack_type == "fgsm":
                adv_pv = fgsm_attack(pv, loss_fn, eps)
            elif attack_type == "pgd":
                adv_pv = pgd_attack(pv, loss_fn, eps, alpha, steps)
            elif attack_type == "veattack":
                adv_pv = veattack(pv, loss_fn, eps, alpha, steps, mode=ve_mode, momentum=ve_momentum)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # ---- Caption (adv) + adv loss (no grads, amp ok) ----
            adv_img = to_pil_from_pixel_values(captioner.processor, adv_pv)
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        adv_caption = captioner.caption(adv_img)
                        adv_loss = float(loss_fn(adv_pv).item())
                else:
                    adv_caption = captioner.caption(adv_img)
                    adv_loss = float(loss_fn(adv_pv).item())

            # ---- Metrics accumulation ----
            loss_increase = adv_loss - orig_loss
            total_loss_increase += loss_increase
            if loss_increase > 0:
                attack_success_count += 1

            # ---- Optional: save image pairs for later image-quality evaluation ----
            if save_root and (max_images_to_save is None or images_saved_count < max_images_to_save):
                img_stem = os.path.splitext(os.path.basename(item["file_name"]))[0]
                orig_path = os.path.join(save_root, f"{img_stem}_orig.png")
                adv_path = os.path.join(save_root, f"{img_stem}_adv.png")
                try:
                    item["image"].save(orig_path, format="PNG")
                    adv_img.save(adv_path, format="PNG")
                    images_saved_count += 1
                except Exception as e:
                    print(f"Warning: failed to save image pair for {img_stem}: {e}")

            # ---- W&B rows (limited) ----
            if len(caption_comparisons) < wandb_display_limit:
                # Defer heavy image conversions: we already have PILs
                clean_img_wandb = wandb.Image(item["image"], caption=f"Original: {orig_caption}")
                adv_img_wandb = wandb.Image(adv_img, caption=f"Adversarial: {adv_caption}")
                caption_comparisons.append([
                    item["image_id"],
                    clean_img_wandb,
                    adv_img_wandb,
                    orig_caption,
                    adv_caption,
                    ref,  # primary reference
                    f"{orig_loss:.3f}",
                    f"{adv_loss:.3f}",
                    f"{loss_increase:.3f}",
                    "Yes" if loss_increase > 0 else "No"
                ])

            # ---- JSONL (for ALL) ----
            rec = {
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "reference_used": ref,
                "hypothesis_orig": orig_caption,
                "hypothesis_adv": adv_caption,
                "all_references": item["captions"],
                "attack": attack_type,
                "epsilon": config["epsilon"],
                "orig_loss": orig_loss,
                "adv_loss": adv_loss,
                "loss_increase": loss_increase,
            }
            if attack_type in ("pgd", "veattack"):
                rec.update({"alpha": config.get("alpha"), "steps": steps})
            if attack_type == "veattack":
                rec.update({"ve_mode": ve_mode, "ve_momentum": ve_momentum})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- Final W&B logs ----
    n = len(dataset)
    attack_success_rate = attack_success_count / max(1, n)
    avg_loss_increase = total_loss_increase / max(1, n)

    wandb.log({
        "attack_success_rate": attack_success_rate,
        "avg_loss_increase": avg_loss_increase,
    })
    
    wandb.log({
        "attack_results": wandb.Table(
            columns=[
                "ID", "Clean_Image", "Adv_Image", "Clean_Caption",
                "Adv_Caption", "Reference", "Loss_Clean", "Loss_Adv",
                "Loss_Increase", "Attack_Success"
            ],
            data=caption_comparisons
        )
    })

    print(f"Caption comparisons logged to W&B: {len(caption_comparisons)} (limit {wandb_display_limit})")
    print(f"JSONL written for all {n} items to: {output_file}")

    return output_file



def evaluate_experiment(baseline_file: str, attack_file: str, images_root: str, config: dict):
    """Evaluate attack results against baseline. Full-corpus metrics; W&B per-image rows limited."""
    
    # Use orjson if available (faster); fall back to json
    try:
        import orjson as _json
        def _loads(b): return _json.loads(b)
    except Exception:
        import json as _json
        def _loads(b): return _json.loads(b)

    wandb_display_limit = int(
        config.get("wandb_display_limit")
        or (config.get("save_images", {}) or {}).get("max_count", 10)
        or 10
    )

    def load_jsonl(path):
        data = []
        with open(path, "rb") as f:  # rb for orjson speed
            for line in f:
                if line.strip():
                    data.append(_loads(line))
        return data

    baseline_data = load_jsonl(baseline_file)
    attack_data = load_jsonl(attack_file)

    # Align by filename (dict lookup O(1))
    ref_map = {b["file_name"]: b for b in baseline_data}

    refs, hyps_base, hyps_adv, img_paths = [], [], [], []
    for a in attack_data:
        b = ref_map.get(a["file_name"])
        if b is None:
            continue
        # Be resilient to key naming
        b_refs = b.get("references") or b.get("all_references") or b.get("captions")
        b_hyp  = b.get("hypothesis") or b.get("hypothesis_orig")
        a_hyp  = a.get("hypothesis_adv") or a.get("hypothesis")
        if not (b_refs and b_hyp and a_hyp):
            continue  # skip malformed
        refs.append(b_refs)
        hyps_base.append(b_hyp)
        hyps_adv.append(a_hyp)
        img_paths.append(os.path.join(images_root, a["file_name"]))

    total = len(hyps_adv)
    print(f"Aligned examples: {total}")

    # ---- Global metrics over ALL examples ----
    bleu_base = compute_bleu(refs, hyps_base)
    bleu_adv  = compute_bleu(refs, hyps_adv)
    rouge_base = compute_rougeL(refs, hyps_base)
    rouge_adv  = compute_rougeL(refs, hyps_adv)

    try:
        # If your compute_clipscore supports batching, expose a config like config["clip_bs"]
        clip_base = compute_clipscore(hyps_base, img_paths)
        clip_adv  = compute_clipscore(hyps_adv, img_paths)
    except Exception as e:
        print(f"CLIPScore failed: {e}")
        clip_base = clip_adv = float("nan")

    bleu_delta  = bleu_adv  - bleu_base
    rouge_delta = rouge_adv - rouge_base
    clip_delta  = clip_adv  - clip_base

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
        "aligned_examples": total,
    }

    # ---- Summary metrics (single point) ----
    experiment_name = f"{config.get('attack','unknown')}_eps{config.get('epsilon','N/A')}"
    if config.get('attack') in ('pgd', 'veattack'):
        experiment_name += f"_alpha{config.get('alpha','N/A')}_steps{config.get('steps','N/A')}"

    wandb.log({
        "experiment_summary_metrics": {
            "experiment": experiment_name,
            "attack_success_rate": wandb.run.summary.get("attack_success_rate", 0),
            "bleu_degradation": abs(bleu_delta),
            "rouge_degradation": abs(rouge_delta),
            "clip_degradation": abs(clip_delta) if not math.isnan(clip_delta) else 0,
            "avg_loss_increase": wandb.run.summary.get("avg_loss_increase", 0),
        }
    })

    # ---- Console print (optional) ----
    print("\n== Metrics ==")
    print(f"BLEU  - baseline: {bleu_base:.4f}  | adversarial: {bleu_adv:.4f} | delta: {bleu_delta:.4f}")
    print(f"ROUGE - baseline: {rouge_base:.4f} | adversarial: {rouge_adv:.4f} | delta: {rouge_delta:.4f}")
    print(f"CLIP  - baseline: {clip_base:.4f}  | adversarial: {clip_adv:.4f} | delta: {clip_delta:.4f}")

    # ---- LIMITED per-image table (lazy image I/O + per-item metrics) ----
    eval_comparisons = []
    limit = min(wandb_display_limit, total)

    for i in range(limit):
        # Lazy image load; if missing, still add a row without the image
        try:
            img = Image.open(img_paths[i])
            wandb_img = wandb.Image(img, caption=f"Image {i+1}")
        except Exception:
            wandb_img = None

        # Per-item metrics: compute only for shown rows
        individual_bleu_base = compute_bleu([refs[i]], [hyps_base[i]])
        individual_bleu_adv  = compute_bleu([refs[i]], [hyps_adv[i]])
        individual_rouge_base = compute_rougeL([refs[i]], [hyps_base[i]])
        individual_rouge_adv  = compute_rougeL([refs[i]], [hyps_adv[i]])

        # Keep references compact
        short_refs = refs[i][:2] if isinstance(refs[i], (list, tuple)) else [str(refs[i])]

        eval_comparisons.append([
            wandb_img,
            hyps_base[i],
            hyps_adv[i],
            str(short_refs),
            f"{individual_bleu_base:.3f}",
            f"{individual_bleu_adv:.3f}",
            f"{(individual_bleu_adv - individual_bleu_base):.3f}",
            f"{individual_rouge_base:.3f}",
            f"{individual_rouge_adv:.3f}",
            f"{(individual_rouge_adv - individual_rouge_base):.3f}",
        ])

    wandb.log({
        "evaluation_results": wandb.Table(
            columns=[
                "image", "baseline_caption", "adversarial_caption",
                "references", "bleu_baseline", "bleu_adversarial", "bleu_delta",
                "rouge_baseline", "rouge_adversarial", "rouge_delta"
            ],
            data=eval_comparisons
        ),
        "evaluation_rows_logged": limit,
    })

    # ---- Image quality metrics (LPIPS/SSIM/PSNR) if saved pairs exist ----
    try:
        pairs_dir = None
        if config.get("save_dir"):
            # Scan the exact run directory (no nested attack subfolder)
            pairs_dir = config["save_dir"]
        if pairs_dir and os.path.isdir(pairs_dir):
            def _psnr(img1_u8: np.ndarray, img2_u8: np.ndarray) -> float:
                mse = float(np.mean((img1_u8.astype(np.float32) - img2_u8.astype(np.float32)) ** 2))
                if mse == 0:
                    return float("inf")
                return 20.0 * math.log10(255.0 / math.sqrt(mse))

            advs = sorted(glob.glob(os.path.join(pairs_dir, "*_adv.png")))
            pairs = []
            for ap in advs:
                op = ap.replace("_adv.png", "_orig.png")
                if os.path.exists(op):
                    pairs.append((op, ap))

            if pairs:
                loss_fn = lpips.LPIPS(net="alex").eval()
                if torch.cuda.is_available():
                    loss_fn = loss_fn.cuda()
                to_tensor = T.ToTensor()

                lpips_vals, ssim_vals, psnr_vals = [], [], []
                resized_count = 0

                for op, ap in pairs:
                    o_pil = Image.open(op).convert("RGB")
                    a_pil = Image.open(ap).convert("RGB")
                    if o_pil.size != a_pil.size:
                        o_pil = o_pil.resize(a_pil.size, Image.BICUBIC)
                        resized_count += 1

                    im_o = to_tensor(o_pil).mul_(2.0).sub_(1.0)
                    im_a = to_tensor(a_pil).mul_(2.0).sub_(1.0)
                    if torch.cuda.is_available():
                        im_o = im_o.cuda(non_blocking=True)
                        im_a = im_a.cuda(non_blocking=True)
                    with torch.no_grad():
                        lp = loss_fn(im_o.unsqueeze(0), im_a.unsqueeze(0)).item()
                    lpips_vals.append(lp)

                    o8 = np.array(o_pil)
                    a8 = np.array(a_pil)
                    ssim_vals.append(ssim(o8, a8, channel_axis=2, data_range=255))
                    psnr_vals.append(_psnr(o8, a8))

                if lpips_vals:
                    wandb.log({
                        "lpips_mean": sum(lpips_vals)/len(lpips_vals),
                        "ssim_mean": sum(ssim_vals)/len(ssim_vals),
                        "psnr_mean": sum(psnr_vals)/len(psnr_vals),
                        "image_quality_pairs": len(lpips_vals),
                        "image_quality_resized_count": resized_count,
                    })
    except Exception as e:
        print(f"Image quality metrics skipped: {e}")

    # ---- Overall experiment summary table (single row) ----
    experiment_summary = [[
        config.get("attack", "unknown"),
        config.get("epsilon", "N/A"),
        config.get("alpha", "N/A") if config.get("attack") in ("pgd", "veattack") else "N/A",
        config.get("steps", "N/A") if config.get("attack") in ("pgd", "veattack") else "N/A",
        total,
        f"{bleu_base:.4f}",
        f"{bleu_adv:.4f}",
        f"{bleu_delta:.4f}",
        f"{rouge_base:.4f}",
        f"{rouge_adv:.4f}",
        f"{rouge_delta:.4f}",
        f"{clip_base:.4f}" if not math.isnan(clip_base) else "N/A",
        f"{clip_adv:.4f}" if not math.isnan(clip_adv) else "N/A",
        f"{clip_delta:.4f}" if not math.isnan(clip_delta) else "N/A",
    ]]

    wandb.log({
        "experiment_summary_table": wandb.Table(
            columns=[
                "attack_type", "eps", "alpha", "steps", "total_images",
                "bleu_baseline", "bleu_adversarial", "bleu_delta",
                "rouge_baseline", "rouge_adversarial", "rouge_delta",
                "clip_baseline", "clip_adversarial", "clip_delta"
            ],
            data=experiment_summary
        )
    })

    return metrics



def main():
    parser = argparse.ArgumentParser(description="Run adversarial attack experiments with config files")
    parser.add_argument("--config", required=True, help="Path to config file")
    
    # Mutually exclusive group for experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument("--experiment", help="Specific experiment name from config")
    exp_group.add_argument("--all", action="store_true", help="Run all experiments defined in config")
    
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after attack")
    parser.add_argument("--device", help="Override device (cuda/cpu)")
    parser.add_argument("--save-images", type=int, metavar="N", help="Save only first N image pairs (overrides config)")
    parser.add_argument("--no-save-images", action="store_true", help="Disable image saving entirely")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.all:
        # Run all experiments
        experiment_names = list(config["experiments"].keys())
        print(f"Running all experiments: {experiment_names}")
        
        for exp_name in experiment_names:
            print(f"\n{'='*50}")
            print(f"Starting experiment: {exp_name}")
            print(f"{'='*50}")
            
            exp_config = get_experiment_config(config, exp_name)
            # Determine run_mode if not specified: based on config filename
            if not exp_config.get("run_mode"):
                cfg_stem = Path(args.config).stem.lower()
                exp_config["run_mode"] = "high_impact" if "high_impact" in cfg_stem else "standard"
            
            # Override device if specified
            if args.device:
                exp_config["model"]["device"] = args.device
            
            print(f"Config: {exp_config}")
            
            # Initialize wandb for this experiment
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                tags=config["wandb"]["tags"] + [exp_name],
                config=exp_config,
                name=f"{exp_name}_{exp_config.get('epsilon', 'baseline')}",
                reinit=True  # Allow multiple wandb runs in same process
            )
            
            try:
                # Load dataset and model for this experiment
                dataset = CocoValDataset(
                    exp_config["data"]["root"],
                    exp_config["data"]["split"],
                    exp_config["data"]["max_images"]
                )
                captioner = create_captioner(exp_config["model"]["name"], exp_config["model"]["device"])
                # Optional generation controls from config
                mcfg = exp_config.get("model", {})
                if hasattr(captioner, "single_sentence") and mcfg.get("single_sentence") is not None:
                    captioner.single_sentence = bool(mcfg.get("single_sentence"))
                if hasattr(captioner, "max_length") and mcfg.get("max_length") is not None:
                    try:
                        captioner.max_length = int(mcfg.get("max_length"))
                    except Exception:
                        pass
                if hasattr(captioner, "prompt") and mcfg.get("prompt"):
                    captioner.prompt = str(mcfg.get("prompt"))

                if exp_config["type"] == "baseline":
                    output_file = run_baseline(exp_config, captioner, dataset)
                    print(f"Baseline results saved to: {output_file}")
                elif exp_config["type"] == "attack":
                    output_file = run_attack(exp_config, captioner, dataset)
                    print(f"Attack results saved to: {output_file}")
                    
                    # Auto-evaluate if requested
                    if args.evaluate:
                        # Use global baseline under eval/
                        baseline_file = os.path.join("eval", "baseline_1000.jsonl")
                        if os.path.exists(baseline_file):
                            print(f"\nEvaluating against baseline: {baseline_file}")
                            evaluate_experiment(
                                baseline_file, 
                                output_file, 
                                os.path.join(exp_config["data"]["root"], exp_config["data"]["split"]),
                                exp_config
                            )
                        else:
                            print(f"Warning: Baseline file {baseline_file} not found. Skipping evaluation.")
                else:
                    raise ValueError(f"Unknown experiment type: {exp_config['type']}")
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")
            finally:
                wandb.finish()
            
            print(f"Completed experiment: {exp_name}")
    else:
        # Run single experiment
        exp_config = get_experiment_config(config, args.experiment)
        # Determine run_mode if not specified: based on config filename
        if not exp_config.get("run_mode"):
            cfg_stem = Path(args.config).stem.lower()
            exp_config["run_mode"] = "high_impact" if "high_impact" in cfg_stem else "standard"
        
        # Override device if specified
        if args.device:
            exp_config["model"]["device"] = args.device
        
        print(f"Running experiment: {args.experiment}")
        print(f"Config: {exp_config}")
        
        # Initialize wandb
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            tags=config["wandb"]["tags"] + [args.experiment],
            config=exp_config,
            name=f"{args.experiment}_{exp_config.get('epsilon', 'baseline')}"
        )
        
        # Load dataset and model
        dataset = CocoValDataset(
            exp_config["data"]["root"],
            exp_config["data"]["split"],
            exp_config["data"]["max_images"]
        )
        captioner = create_captioner(exp_config["model"]["name"], exp_config["model"]["device"])
        # Optional generation controls from config
        mcfg = exp_config.get("model", {})
        if hasattr(captioner, "single_sentence") and mcfg.get("single_sentence") is not None:
            captioner.single_sentence = bool(mcfg.get("single_sentence"))
        if hasattr(captioner, "max_length") and mcfg.get("max_length") is not None:
            try:
                captioner.max_length = int(mcfg.get("max_length"))
            except Exception:
                pass
        if hasattr(captioner, "prompt") and mcfg.get("prompt"):
            captioner.prompt = str(mcfg.get("prompt"))

        if exp_config["type"] == "baseline":
            output_file = run_baseline(exp_config, captioner, dataset)
            print(f"Baseline results saved to: {output_file}")
        elif exp_config["type"] == "attack":
            output_file = run_attack(exp_config, captioner, dataset)
            print(f"Attack results saved to: {output_file}")
            
            # Auto-evaluate if requested
            if args.evaluate:
                baseline_file = os.path.join("eval", "baseline_1000.jsonl")
                if os.path.exists(baseline_file):
                    print(f"\nEvaluating against baseline: {baseline_file}")
                    evaluate_experiment(
                        baseline_file, 
                        output_file, 
                        os.path.join(exp_config["data"]["root"], exp_config["data"]["split"]),
                        exp_config
                    )
                else:
                    print(f"Warning: Baseline file {baseline_file} not found. Skipping evaluation.")
        else:
            raise ValueError(f"Unknown experiment type: {exp_config['type']}")
        
        wandb.finish()


if __name__ == "__main__":
    main()
