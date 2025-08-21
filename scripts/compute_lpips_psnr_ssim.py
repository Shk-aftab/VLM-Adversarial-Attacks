# scripts/compute_lpips_psnr_ssim.py
import argparse
import glob
import math
import os

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import lpips
from skimage.metrics import structural_similarity as ssim
import wandb


def psnr(img1_u8: np.ndarray, img2_u8: np.ndarray) -> float:
    """PSNR in dB for two uint8 images shaped [H,W,C]."""
    mse = np.mean((img1_u8.astype(np.float32) - img2_u8.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def load_pair_paths(root: str, limit: int | None = None) -> list[tuple[str, str]]:
    """
    Find pairs (orig, adv) by scanning *_adv.png and mapping to *_orig.png.
    Returns list of (orig_path, adv_path).
    """
    advs = sorted(glob.glob(os.path.join(root, "*_adv.png")))
    pairs = []
    for ap in advs:
        op = ap.replace("_adv.png", "_orig.png")
        if os.path.exists(op):
            pairs.append((op, ap))
        # else: silently skip if orig missing
        if limit and len(pairs) >= limit:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Compute LPIPS/SSIM/PSNR over saved *_orig.png / *_adv.png pairs."
    )
    ap.add_argument("--dir", required=True, help="Folder with PNG pairs (e.g., saved_adv/fgsm)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on #pairs")
    ap.add_argument("--wandb", action="store_true", help="Log results to wandb")
    ap.add_argument("--wandb_run_name", help="Name for wandb run")
    args = ap.parse_args()
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="adversarial-blip-attacks",
            name=args.wandb_run_name or "image_quality_metrics",
            tags=["lpips", "ssim", "psnr", "image_quality"]
        )

    pairs = load_pair_paths(args.dir, args.limit)
    if not pairs:
        print("No pairs found. Make sure *_orig.png and *_adv.png exist in:", args.dir)
        return

    # LPIPS model (alex). Uses GPU if available.
    loss_fn = lpips.LPIPS(net="alex").eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    to_tensor = T.ToTensor()  # -> float [0,1], CHW

    lpips_vals = []
    ssim_vals = []
    psnr_vals = []
    resized_count = 0

    for orig_path, adv_path in pairs:
        # Load images
        o_pil = Image.open(orig_path).convert("RGB")
        a_pil = Image.open(adv_path).convert("RGB")

        # Ensure same size: resize orig to match adv if needed (adv reflects model input size)
        if o_pil.size != a_pil.size:
            o_pil = o_pil.resize(a_pil.size, Image.BICUBIC)
            resized_count += 1

        # ----- LPIPS (expects tensors in [-1, 1]) -----
        im_o = to_tensor(o_pil).mul_(2.0).sub_(1.0)  # [-1,1], CHW
        im_a = to_tensor(a_pil).mul_(2.0).sub_(1.0)
        if torch.cuda.is_available():
            im_o = im_o.cuda(non_blocking=True)
            im_a = im_a.cuda(non_blocking=True)
        with torch.no_grad():
            lp = loss_fn(im_o.unsqueeze(0), im_a.unsqueeze(0)).item()
        lpips_vals.append(lp)

        # ----- SSIM / PSNR on uint8 (post-resize) -----
        o8 = np.array(o_pil)  # HWC, uint8
        a8 = np.array(a_pil)
        ssim_vals.append(ssim(o8, a8, channel_axis=2, data_range=255))
        psnr_vals.append(psnr(o8, a8))

    n = len(lpips_vals)
    lpips_mean = sum(lpips_vals)/n
    ssim_mean = sum(ssim_vals)/n
    psnr_mean = sum(psnr_vals)/n
    
    print(f"Pairs evaluated: {n}")
    if resized_count:
        print(f"Note: resized {resized_count}/{n} orig images to match adv size.")

    print(f"LPIPS (alex) mean: {lpips_mean:.4f}")
    print(f"SSIM        mean: {ssim_mean:.4f}")
    print(f"PSNR         mean: {psnr_mean:.2f} dB")
    
    # Log to wandb if enabled
    if args.wandb:
        wandb.log({
            "lpips_mean": lpips_mean,
            "ssim_mean": ssim_mean,
            "psnr_mean": psnr_mean,
            "pairs_evaluated": n,
            "resized_count": resized_count
        })
        wandb.finish()
    
    return {
        "lpips_mean": lpips_mean,
        "ssim_mean": ssim_mean,
        "psnr_mean": psnr_mean,
        "pairs_evaluated": n
    }


if __name__ == "__main__":
    main()
