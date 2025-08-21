# scripts/attack_and_eval.py
import argparse
import json
import math
import os
from tqdm import tqdm
import torch
import wandb

from src.utils.data import CocoValDataset
from src.model.blip_captioner import BLIPCaptioner
from src.utils.attack_runner import pixel_values_from_image, to_pil_from_pixel_values
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.config import load_config, get_experiment_config, parse_float_maybe_frac
from eval.metrics import compute_bleu, compute_rougeL, compute_clipscore


# Moved to src.utils.config


def main():
    ap = argparse.ArgumentParser(description="Run adversarial attacks (FGSM/PGD) on BLIP and save outputs.")
    # Config-based usage
    ap.add_argument("--config", help="Path to config file (enables config-based mode)")
    ap.add_argument("--experiment", help="Experiment name from config (requires --config)")
    ap.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    
    # Legacy CLI arguments (for backward compatibility)
    ap.add_argument("--data_root", help="Root folder of COCO 2017 (e.g., data/coco2017)")
    ap.add_argument("--split", default="val2017", help="Dataset split folder name (default: val2017)")
    ap.add_argument("--model_name", default="Salesforce/blip-image-captioning-large", help="HF model id")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (auto-falls back to cpu if no GPU)")
    ap.add_argument("--attack", choices=["fgsm", "pgd"], help="Attack type")
    ap.add_argument("--epsilon", type=str, default="4/255", help="Perturbation budget (e.g., 4/255)")
    ap.add_argument("--alpha", type=str, default="1/255", help="PGD step size (ignored for FGSM)")
    ap.add_argument("--steps", type=int, default=10, help="PGD steps (ignored for FGSM)")
    ap.add_argument("--out", help="Path to write JSONL results")
    ap.add_argument("--max_images", type=int, default=None, help="Limit #images (for quick runs)")
    ap.add_argument("--save_dir", type=str, default="saved_adv", help="Folder to save adv/orig PNGs")
    ap.add_argument("--no_save", action="store_true", help="If set, do not save PNG images")
    args = ap.parse_args()
    
    # Determine mode: config-based or legacy CLI
    if args.config and args.experiment:
        # Config-based mode
        config = load_config(args.config)
        exp_config = get_experiment_config(config, args.experiment)
        
        # Initialize wandb if requested
        if args.wandb:
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                name=args.experiment,
                tags=config["wandb"]["tags"] + [exp_config.get("attack", "baseline")],
                config=exp_config
            )
        
        # Extract parameters from config
        data_root = exp_config["data"]["root"]
        split = exp_config["data"]["split"]
        model_name = exp_config["model"]["name"]
        device = exp_config["model"]["device"]
        attack = exp_config.get("attack")
        epsilon = exp_config.get("epsilon", "4/255")
        alpha = exp_config.get("alpha", "1/255")
        steps = exp_config.get("steps", 10)
        out = exp_config["output_file"]
        max_images = exp_config["data"]["max_images"]
        save_dir = exp_config.get("save_dir", "saved_adv")
        no_save = False
        
    else:
        # Legacy CLI mode
        if not args.data_root or not args.attack or not args.out:
            print("Error: In legacy mode, --data_root, --attack, and --out are required")
            return
        
        data_root = args.data_root
        split = args.split
        model_name = args.model_name
        device = args.device
        attack = args.attack
        epsilon = args.epsilon
        alpha = args.alpha
        steps = args.steps
        out = args.out
        max_images = args.max_images
        save_dir = args.save_dir
        no_save = args.no_save

    # Resolve device
    device = "cuda" if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"

    # Dataset & model
    ds = CocoValDataset(data_root, split, max_images)
    cap = BLIPCaptioner(model_name, device)

    # Attack params
    eps = parse_float_maybe_frac(epsilon)
    alpha = parse_float_maybe_frac(alpha)

    # Where to save images
    save_root = os.path.join(save_dir, attack)
    if not no_save:
        os.makedirs(save_root, exist_ok=True)

    # Results file
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    # Image saving configuration
    save_images_enabled = True  # Legacy mode always saves by default
    max_images_to_save = None   # Save all by default in legacy mode
    
    # Track metrics for wandb
    attack_success_count = 0
    total_loss_increase = 0.0
    images_saved_count = 0
    
    # Collect caption comparisons for wandb table
    caption_comparisons = []

    with open(out, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(ds, total=len(ds))):
            # one reference caption for untargeted loss maximization
            ref = item["captions"][0]

            # normalized pixel_values via the processor
            pv = pixel_values_from_image(cap.processor, item["image"], device)
            
            # Get original caption and loss for tracking
            orig_caption = cap.caption(item["image"])

            # differentiable loss wrt pixel_values using the chosen reference
            def loss_fn_wrapper(x):
                ti = cap.processor(text=ref, return_tensors="pt").to(device)
                outputs = cap.model(
                    pixel_values=x,
                    input_ids=ti["input_ids"],
                    attention_mask=ti["attention_mask"],
                    labels=ti["input_ids"],
                )
                return outputs.loss
            
            # Record original loss
            with torch.no_grad():
                orig_loss = loss_fn_wrapper(pv).item()

            # run attack
            if attack == "fgsm":
                adv_pv = fgsm_attack(pv, loss_fn_wrapper, eps)
            else:
                adv_pv = pgd_attack(pv, loss_fn_wrapper, eps, alpha, steps)

            # captions on adversarial image
            adv_img = to_pil_from_pixel_values(cap.processor, adv_pv)
            hyp_adv = cap.caption(adv_img)
            
            # Record adversarial loss
            with torch.no_grad():
                adv_loss = loss_fn_wrapper(adv_pv).item()
            
            # Track attack success
            loss_increase = adv_loss - orig_loss
            total_loss_increase += loss_increase
            if loss_increase > 0:
                attack_success_count += 1

            # save PNGs (orig & adv) with stable names
            should_save_image = (
                not no_save and 
                save_images_enabled and 
                (max_images_to_save is None or images_saved_count < max_images_to_save)
            )
            
            if should_save_image:
                # original PIL is already available from dataset iterator
                orig_img = item["image"]

                # derive stem from original file name (e.g., 000000123456.jpg -> 000000123456)
                img_name = item["file_name"]
                img_stem = os.path.splitext(os.path.basename(img_name))[0]

                orig_path = os.path.join(save_root, f"{img_stem}_orig.png")
                adv_path = os.path.join(save_root, f"{img_stem}_adv.png")

                # lossless PNG for LPIPS/visuals
                orig_img.save(orig_path, format="PNG")
                adv_img.save(adv_path, format="PNG")
                images_saved_count += 1

            # Collect data for wandb table (limit to first 30 for readability)
            if len(caption_comparisons) < 30:
                # Compute individual metrics for this image
                img_path = os.path.join(data_root, split, item["file_name"])
                
                # Individual BLEU scores
                bleu_orig = compute_bleu([item["captions"]], [orig_caption])
                bleu_adv = compute_bleu([item["captions"]], [hyp_adv])
                
                # Individual ROUGE scores
                rouge_orig = compute_rougeL([item["captions"]], [orig_caption])
                rouge_adv = compute_rougeL([item["captions"]], [hyp_adv])
                
                # Individual CLIP scores (with error handling)
                try:
                    clip_orig = compute_clipscore([orig_caption], [img_path])
                    clip_adv = compute_clipscore([hyp_adv], [img_path])
                except:
                    clip_orig = clip_adv = float("nan")
                
                # Create wandb images
                clean_img_wandb = wandb.Image(item["image"], caption=f"Original: {orig_caption}")
                adv_img_wandb = wandb.Image(adv_img, caption=f"Adversarial: {hyp_adv}")
                
                caption_comparisons.append([
                    item["image_id"],
                    epsilon,
                    clean_img_wandb,
                    adv_img_wandb,
                    orig_caption,
                    hyp_adv,
                    item["captions"][0],  # Primary reference
                    f"{bleu_orig:.3f}",
                    f"{bleu_adv:.3f}",
                    f"{rouge_orig:.3f}",
                    f"{rouge_adv:.3f}",
                    f"{clip_orig:.3f}" if not math.isnan(clip_orig) else "N/A",
                    f"{clip_adv:.3f}" if not math.isnan(clip_adv) else "N/A",
                    f"{orig_loss:.4f}",
                    f"{adv_loss:.4f}",
                    f"{loss_increase:.4f}",
                    "✅" if loss_increase > 0 else "❌"
                ])
            
            # write one JSON line per example
            rec = {
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "reference_used": ref,
                "hypothesis_orig": orig_caption,
                "hypothesis_adv": hyp_adv,
                "all_references": item["captions"],
                "attack": attack,
                "epsilon": epsilon,
                "orig_loss": orig_loss,
                "adv_loss": adv_loss,
                "loss_increase": loss_increase,
                **({"alpha": alpha, "steps": steps} if attack == "pgd" else {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Log final statistics to wandb if enabled
    if args.wandb or (args.config and args.experiment):
        attack_success_rate = attack_success_count / len(ds)
        avg_loss_increase = total_loss_increase / len(ds)
        
        wandb.log({
            "attack_success_rate": attack_success_rate,
            "avg_loss_increase": avg_loss_increase,
            "total_images": len(ds),
            "attack_type": attack,
            "epsilon_value": eps,
            "images_saved": images_saved_count
        })
        
        # Log comprehensive results table
        wandb.log({
            "attack_results": wandb.Table(
                columns=["ID", "Epsilon", "Clean_Image", "Adv_Image", "Clean_Caption", 
                        "Adv_Caption", "Reference", "BLEU_Clean", "BLEU_Adv", 
                        "ROUGE_Clean", "ROUGE_Adv", "CLIP_Clean", "CLIP_Adv",
                        "Loss_Clean", "Loss_Adv", "Loss_Increase", "Attack_Success"],
                data=caption_comparisons
            )
        })
        
        print(f"Images saved: {images_saved_count}/{len(ds)}")
        print(f"Caption comparisons logged: {len(caption_comparisons)}")


if __name__ == "__main__":
    main()
