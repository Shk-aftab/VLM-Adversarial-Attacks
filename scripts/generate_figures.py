#!/usr/bin/env python3
"""
Generate figures for adversarial BLIP experiments.

Inputs (auto-detected if present):
- Root-level: All_experiment_summary_table.csv, All_experiment_summary_table_high_impact.csv
- Folders:
  - result/standard/: bleu_degradation.csv, rouge_degradation.csv, clip_degradation.csv, avg_loss_increase.csv
  - result/high_impact/: same

Outputs:
- result/figures/standard/*.png
- result/figures/high_impact/*.png
"""

import os
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Parsing helpers
# -----------------------

EPS_FRAC_RE = re.compile(r"(?P<num>\d+)\s*/\s*255")

def parse_eps_to_float(eps_str: str) -> float:
    if isinstance(eps_str, (int, float)):
        return float(eps_str)
    s = str(eps_str).strip().strip('"').strip("'")
    m = EPS_FRAC_RE.match(s)
    if m:
        return float(m.group("num")) / 255.0
    try:
        return float(s)
    except Exception:
        # Fallback: try to extract the leading number
        nums = re.findall(r"[\d.]+", s)
        return float(nums[0]) if nums else float("nan")

def parse_label(label: str) -> dict:
    """
    Parse labels like:
      - pgd_eps8_s20_a1_8/255
      - pgd_eps8_8/255
      - fgsm_eps10_10/255
    Return dict: {attack_type, eps_str, eps_float, alpha_str?, steps?}
    """
    lab = str(label).strip()
    out = {"attack_type": None, "eps_str": None, "eps_float": float("nan"),
           "alpha_str": None, "steps": None}
    if lab.startswith("pgd"):
        out["attack_type"] = "pgd"
    elif lab.startswith("fgsm"):
        out["attack_type"] = "fgsm"
    elif lab.startswith("veattack"):
        out["attack_type"] = "veattack"
    else:
        out["attack_type"] = "unknown"

    # Extract eps fraction (last token like 8/255 or 10/255)
    eps_match = re.findall(r"(\d+/\s*255)", lab)
    if eps_match:
        out["eps_str"] = eps_match[-1].replace(" ", "")
        out["eps_float"] = parse_eps_to_float(out["eps_str"])

    # Extract steps and alpha if present: _s<steps>_a<alpha_num>/255
    s_match = re.search(r"_s(\d+)", lab)
    if s_match:
        out["steps"] = int(s_match.group(1))
    a_match = re.search(r"_a(\d+)\s*/\s*255", lab)
    if a_match:
        out["alpha_str"] = f"{a_match.group(1)}/255"

    return out

# -----------------------
# Loaders
# -----------------------

def load_from_aggregate(path: str) -> pd.DataFrame:
    """
    Load aggregate table with columns including:
      attack_type, eps, alpha, steps, bleu_delta, rouge_delta, clip_delta, ...
    Returns DataFrame with normalized numeric columns.
    """
    df = pd.read_csv(path)
    # Normalize columns
    if "eps" in df.columns:
        df["eps_str"] = df["eps"].astype(str)
        df["eps_float"] = df["eps_str"].map(parse_eps_to_float)
    else:
        df["eps_str"] = None
        df["eps_float"] = float("nan")

    if "alpha" in df.columns:
        df["alpha_str"] = df["alpha"].astype(str)
    else:
        df["alpha_str"] = None

    if "steps" in df.columns:
        # Some CSVs may have "N/A" for FGSM
        def _to_int_or_na(x):
            try:
                return int(x)
            except Exception:
                return None
        df["steps"] = df["steps"].map(_to_int_or_na)
    else:
        df["steps"] = None

    # Deltas columns may be named *_delta
    # Ensure BLEU/ROUGE/CLIP delta columns exist (float)
    for col in ["bleu_delta", "rouge_delta", "clip_delta"]:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Avg loss increase present? If not, leave NaN
    if "avg_loss_increase" in df.columns:
        df["avg_loss_increase"] = pd.to_numeric(df["avg_loss_increase"], errors="coerce")
    else:
        df["avg_loss_increase"] = float("nan")

    # attack_type string cleanup
    df["attack_type"] = df["attack_type"].astype(str).str.lower().str.strip()
    return df

def load_from_metric_csvs(folder: str) -> pd.DataFrame:
    """
    Merge per-metric CSVs of the form:
      label, value
    for BLEU/ROUGE/CLIP degradation and avg_loss_increase.
    """
    def _load_one(filename: str, value_col: str) -> pd.DataFrame:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["label", value_col])
        # Some files may contain trailing dots/format quirks; be tolerant
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip(".")
                if not line:
                    continue
                # Split only on first comma
                if "," in line:
                    key, val = line.split(",", 1)
                    try:
                        rows.append((key.strip(), float(val)))
                    except Exception:
                        # Skip malformed
                        continue
        return pd.DataFrame(rows, columns=["label", value_col])

    bleu = _load_one("bleu_degradation.csv", "bleu_delta")
    rouge = _load_one("rouge_degradation.csv", "rouge_delta")
    clip = _load_one("clip_degradation.csv", "clip_delta")
    loss = _load_one("avg_loss_increase.csv", "avg_loss_increase")

    # Merge on label
    df = bleu.merge(rouge, on="label", how="outer") \
             .merge(clip, on="label", how="outer") \
             .merge(loss, on="label", how="outer")

    # Parse label to metadata
    meta = df["label"].apply(parse_label).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)

    # Normalize types
    df["bleu_delta"] = pd.to_numeric(df["bleu_delta"], errors="coerce")
    df["rouge_delta"] = pd.to_numeric(df["rouge_delta"], errors="coerce")
    df["clip_delta"] = pd.to_numeric(df["clip_delta"], errors="coerce")
    df["avg_loss_increase"] = pd.to_numeric(df["avg_loss_increase"], errors="coerce")
    return df

def load_setting(
    aggregate_path: str | None,
    folder_fallback: str | None
) -> pd.DataFrame:
    if aggregate_path and os.path.exists(aggregate_path):
        return load_from_aggregate(aggregate_path)
    if folder_fallback and os.path.isdir(folder_fallback):
        return load_from_metric_csvs(folder_fallback)
    return pd.DataFrame()

# -----------------------
# Plotters
# -----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_bar_metric(df: pd.DataFrame, metric: str, out_path: str, title: str):
    if df.empty or metric not in df.columns:
        return
    d = df.copy()
    d = d.dropna(subset=[metric, "attack_type", "eps_float"])
    # Order by severity (most negative first)
    d = d.sort_values(metric)
    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    sns.barplot(data=d, y="label" if "label" in d.columns else "eps_str", x=metric, hue="attack_type", dodge=False)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title(title)
    plt.xlabel(f"{metric} (negative = more damage)")
    plt.ylabel("config")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_pgd_eps_trend(df: pd.DataFrame, metric: str, out_path: str, title: str):
    if df.empty or metric not in df.columns:
        return
    d = df.copy()
    d = d[d["attack_type"] == "pgd"].dropna(subset=["eps_float", metric])
    if d.empty:
        return
    plt.figure(figsize=(8, 5))
    # Use hue=steps, style=alpha_str to reflect settings if available
    sns.scatterplot(data=d, x="eps_float", y=metric, hue="steps", style="alpha_str", s=80)
    sns.lineplot(data=d.sort_values("eps_float"), x="eps_float", y=metric, hue="steps", legend=False, alpha=0.4)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.title(title)
    plt.xlabel("epsilon (as fraction of 1)")
    plt.ylabel(f"{metric} (negative = more damage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_scatter_loss_vs_bleu(df: pd.DataFrame, out_path: str, title: str):
    if df.empty:
        return
    d = df.copy()
    if "avg_loss_increase" not in d.columns or "bleu_delta" not in d.columns:
        return
    d = d.dropna(subset=["avg_loss_increase", "bleu_delta"])
    if d.empty:
        return
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=d, x="avg_loss_increase", y="bleu_delta", hue="attack_type", style="attack_type", s=80)
    # Optional: fit a lowess line? Keep simple: show y=mx trend via regplot without line eq
    sns.regplot(data=d, x="avg_loss_increase", y="bleu_delta", scatter=False, color="gray", lowess=True)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.title(title)
    plt.xlabel("avg_loss_increase (higher = stronger attack)")
    plt.ylabel("BLEU delta (negative = more damage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Generate figures for BLIP adversarial experiments")
    parser.add_argument("--standard-agg", default=os.path.join("result", "standard", "All_experiment_summary_table.csv"))
    parser.add_argument("--high-agg", default=os.path.join("result", "high_impact", "All_experiment_summary_table_high_impact.csv"))
    parser.add_argument("--standard-dir", default=os.path.join("result", "standard"))
    parser.add_argument("--high-dir", default=os.path.join("result", "high_impact"))
    parser.add_argument("--out-standard", default=os.path.join("result", "figures", "standard"))
    parser.add_argument("--out-high", default=os.path.join("result", "figures", "high_impact"))
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    # Load dataframes
    df_std = load_setting(args.standard_agg, args.standard_dir)
    df_hi  = load_setting(args.high_agg, args.high_dir)

    # Ensure output dirs
    ensure_dir(args.out_standard)
    ensure_dir(args.out_high)

    # 1) Bar plots (BLEU/ROUGE) per setting
    plot_bar_metric(df_std, "bleu_delta", os.path.join(args.out_standard, "bleu_bar_standard.png"),
                    "Standard: BLEU degradation by configuration")
    plot_bar_metric(df_std, "rouge_delta", os.path.join(args.out_standard, "rouge_bar_standard.png"),
                    "Standard: ROUGE-L degradation by configuration")

    plot_bar_metric(df_hi, "bleu_delta", os.path.join(args.out_high, "bleu_bar_high_impact.png"),
                    "High-Impact: BLEU degradation by configuration")
    plot_bar_metric(df_hi, "rouge_delta", os.path.join(args.out_high, "rouge_bar_high_impact.png"),
                    "High-Impact: ROUGE-L degradation by configuration")

    # 2) PGD Eps trends (BLEU/ROUGE) per setting
    plot_pgd_eps_trend(df_std, "bleu_delta", os.path.join(args.out_standard, "pgd_bleu_vs_eps_standard.png"),
                       "Standard: PGD BLEU degradation vs epsilon")
    plot_pgd_eps_trend(df_std, "rouge_delta", os.path.join(args.out_standard, "pgd_rouge_vs_eps_standard.png"),
                       "Standard: PGD ROUGE-L degradation vs epsilon")

    plot_pgd_eps_trend(df_hi, "bleu_delta", os.path.join(args.out_high, "pgd_bleu_vs_eps_high_impact.png"),
                       "High-Impact: PGD BLEU degradation vs epsilon")
    plot_pgd_eps_trend(df_hi, "rouge_delta", os.path.join(args.out_high, "pgd_rouge_vs_eps_high_impact.png"),
                       "High-Impact: PGD ROUGE-L degradation vs epsilon")

    # 3) Scatter: avg_loss_increase vs BLEU
    plot_scatter_loss_vs_bleu(df_std, os.path.join(args.out_standard, "loss_vs_bleu_standard.png"),
                              "Standard: Avg loss increase vs BLEU degradation")
    plot_scatter_loss_vs_bleu(df_hi, os.path.join(args.out_high, "loss_vs_bleu_high_impact.png"),
                              "High-Impact: Avg loss increase vs BLEU degradation")

    print(f"Figures written to:\n - {args.out_standard}\n - {args.out_high}")

if __name__ == "__main__":
    main()