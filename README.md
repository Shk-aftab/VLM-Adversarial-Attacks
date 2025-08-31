# Adversarial Attacks on Vision–Language Captioners (BLIP / BLIP‑2 / LLaVA)

## Project Overview

This project investigates the robustness of vision–language captioning models by testing adversarial attacks across BLIP, BLIP‑2, and LLaVA. We target the image pathway to see how small perturbations affect caption quality.

### What We're Testing

**Research Question**: How vulnerable are state-of-the-art image captioning models to adversarial examples, and what is the trade-off between attack strength and image quality degradation?

**Attack Methods**:
- **FGSM (Fast Gradient Sign Method)**: single-step gradient sign
- **PGD (Projected Gradient Descent)**: multi-step iterative with projection
- **VEAttack (Velocity-Enhanced)**: momentum-based variant with mode=max/min

**Target Models (auto-selected via factory)**:
- BLIP (`BlipForConditionalGeneration`)
- BLIP‑2 (`Blip2ForConditionalGeneration`)
- LLaVA (`LlavaForConditionalGeneration`)

**Attack Strategy**: We maximize the negative log-likelihood loss of reference captions, forcing the model to generate captions that deviate significantly from ground truth.

### Evaluation Pipeline

Our evaluation measures both **attack effectiveness** and **perceptual quality**:

**Caption Quality Metrics**:
- **BLEU**: Measures n-gram overlap between generated and reference captions (higher = better)
- **ROUGE-L**: Evaluates longest common subsequence similarity (higher = better)  
- **CLIPScore**: Semantic similarity between generated captions and images using CLIP embeddings (higher = better)

**Image Quality Metrics**:
- **LPIPS**: Learned Perceptual Image Patch Similarity - measures perceptual distance (lower = better)
- **PSNR**: Peak Signal-to-Noise Ratio - measures pixel-level distortion in dB (higher = better)
- **SSIM**: Structural Similarity Index - measures structural similarity (higher = better)

**Attack Success Metrics**:
- **Attack Success Rate**: Percentage of images where loss increased (indicating successful attack)
- **Average Loss Increase**: Mean increase in negative log-likelihood loss

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
bash scripts/download_coco2017_val.sh data/coco2017
```

Notes:
- Hugging Face models will be downloaded on first use (BLIP, BLIP‑2, LLaVA).
- Ensure GPU is available for larger checkpoints (e.g., BLIP‑2 2.7B, LLaVA 7B).

## Configuration

Edit `configs/experiments.yaml` to define your experiments. Key sections:

- **Model settings**: `model.name` (string contains "blip2", "llava", or defaults to BLIP), `model.device` ("cuda"/"cpu")
- **Data settings**: Dataset path, split, max images
- **Wandb settings**: Project name, entity, tags
- **Experiment definitions**: attack configurations and hyperparameters

Example model configs:

```yaml
model:
  name: "Salesforce/blip-image-captioning-large"   # BLIP
  device: "cuda"

# or BLIP-2
model:
  name: "Salesforce/blip2-opt-2.7b"               # BLIP-2
  device: "cuda"

# or LLaVA
model:
  name: "llava-hf/llava-1.5-7b-hf"                # LLaVA
  device: "cuda"
```

### Experiment Types

**Baseline**: Uses the pretrained BLIP captioning model to generate captions for unaltered COCO validation images. This establishes ground truth performance metrics that adversarial attacks are compared against.

**Attack Experiments**: Each experiment tests different hyperparameter combinations:

- **epsilon**: Perturbation budget (e.g., "2/255", "4/255", "8/255") - controls maximum pixel change allowed
- **alpha**: Step size for PGD attacks (e.g., "1/255") - controls how large each iterative step is
- **steps**: Number of iterations for PGD/VEAttack (e.g., 10)
- **ve_mode**: `max` or `min` (for VEAttack)
- **ve_momentum**: momentum coefficient (e.g., 0.9) for VEAttack

The config defines experiments like `fgsm_eps2`, `pgd_eps4`, etc. where the name indicates the attack type and key hyperparameters for easy identification.

## Usage

### 🚀 Config-Based Runner (Recommended)

#### Command Arguments Explained

**`scripts/run_experiment.py` arguments:**
- `--config`: **Required** - Path to config file (e.g., `configs/experiments.yaml`)
- `--experiment`: **Required (unless using --all)** - Name from config (e.g., `fgsm_eps4`, `baseline`)
- `--all`: **Optional** - Run all experiments in the config
- `--evaluate`: **Optional** - Runs evaluation after attack
- `--baseline`: deprecated; baseline path is read from config in evaluation section
- `--save-images N`: **Optional** - Save only first N image pairs (overrides config)
- `--no-save-images`: **Optional** - Disable image saving entirely

What happens without `--evaluate`:
- Runs attack and saves results to JSONL file
- Logs attack success metrics to wandb
- **No caption quality evaluation** (BLEU/ROUGE/CLIPScore)

What happens with `--evaluate`:
- Everything above **PLUS**
- Computes BLEU, ROUGE, CLIPScore comparing baseline vs adversarial captions
- Logs metric deltas and sample comparisons to wandb
- Prints comprehensive evaluation results

#### Usage Examples

Run all configured experiments (baseline + all attacks) with evaluation:

```bash
python scripts/run_experiment.py --config configs/experiments.yaml --all --evaluate
```

Run a specific experiment (example: FGSM eps=4/255) with evaluation:

```bash
python scripts/run_experiment.py --config configs/experiments.yaml --experiment fgsm_eps4 --evaluate
```

1) Run baseline captioning:
```bash
python scripts/run_experiment.py --config configs/experiments.yaml --experiment baseline
```

2) Run FGSM attacks:
```bash
# Attack only (no evaluation)
python scripts/run_experiment.py --config configs/experiments.yaml --experiment fgsm_eps4


3) Run PGD/VEAttack attacks:
```bash
# Different epsilon values with 10 steps, alpha=1/255
python scripts/run_experiment.py --config configs/experiments.yaml --experiment pgd_eps2 --evaluate
```

4) Compute image quality metrics (optional):
```bash
# Saved image pairs are organized by model and run mode.
# Example for BLIP-2 on standard runs:
python scripts/compute_lpips_psnr_ssim.py --dir runs/Salesforce__blip2-opt-2.7b/standard/fgsm_eps4_255_1000
```

### 🔧 Legacy CLI Usage (Backward Compatible)

The original CLI interface still works:

**Baseline:**
```bash
python scripts/caption_images.py --data_root data/coco2017 --out eval/baseline_1000.jsonl --max_images 1000
```

**FGSM Attack:**
```bash
python scripts/attack_and_eval.py --data_root data/coco2017 --attack fgsm --epsilon 2/255 --out eval/fgsm_eps2_255_1000.jsonl --max_images 1000 --save_dir runs/fgsm_eps2_255_1000
```

**Evaluation:**
```bash
python scripts/evaluate_results.py --baseline eval/baseline_1000.jsonl --adversarial eval/fgsm_eps2_255_1000.jsonl --images_root data/coco2017/val2017
```

## Experiment Configuration

The `configs/experiments.yaml` file defines all experiments:

```yaml
experiments:
  baseline:
    type: "baseline"
    # Baseline is stored globally at eval/baseline_1000.jsonl and reused by all runs
    output_file: "eval/baseline_1000.jsonl"
    
  fgsm_eps2:
    type: "attack"
    attack: "fgsm"
    epsilon: "2/255"
    # Attack JSONLs are automatically routed to eval/<model>/<run_mode>/<base_name>
    output_file: "eval/fgsm_eps2_255_1000.jsonl"
    # Saved image pairs are routed to runs/<model>/<run_mode>/<stem>
    save_dir: "runs/fgsm_eps2_255_1000"
    
  pgd_eps2:
    type: "attack"
    attack: "pgd"
    epsilon: "2/255"
    alpha: "1/255"
    steps: 10
    output_file: "eval/pgd_s10_a1_255_eps2_255_1000.jsonl"
    save_dir: "runs/pgd_eps2_255_1000"

### Path Layout (when using scripts/run_experiment.py)

- Baseline JSONL: `eval/baseline_1000.jsonl` (global, reused)
- Attack JSONLs: `eval/<model_name>/<run_mode>/<name>.jsonl`
  - Example: `eval/Salesforce__blip2-opt-2.7b/standard/pgd_s10_a1_255_eps4_255_1000.jsonl`
- Saved image pairs: `runs/<model_name>/<run_mode>/<stem>/`
  - Example: `runs/Salesforce__blip2-opt-2.7b/high_impact/fgsm_eps8_255_1000/`

Notes:
- `<model_name>` is sanitized by replacing `/` with `__`.
- `<run_mode>` is `standard` or `high_impact` (inferred from config filename or set via `run_mode` in the experiment config).

```



