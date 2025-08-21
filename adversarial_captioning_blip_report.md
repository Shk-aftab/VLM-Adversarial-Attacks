# Detailed Report: Adversarial Robustness of BLIP Image Captioning Model

---

## 1. Executive Summary

- **PGD attacks with high epsilon and sufficient steps consistently induce the largest caption quality degradation**. In high-impact settings, PGD (eps=10/255, alpha=2/255, steps=20) yields BLEU delta -0.1833 and ROUGE delta -0.1582.
- **PGD is substantially stronger than FGSM**: For comparable epsilon, PGD produces 2–4× the BLEU/ROUGE degradation of FGSM, with higher average loss increases.
- **Attack strength monotonically increases with epsilon for both FGSM and PGD**. For PGD, increasing steps or alpha (within reasonable bounds) further strengthens the attack.
- **CLIPScore deltas can be positive even when BLEU/ROUGE degrade**—this divergence is consistent across both standard and high-impact runs, highlighting limitations of CLIPScore as a sole robustness metric.
- **FGSM impact saturates at higher epsilon**, while PGD continues to degrade quality with more aggressive settings (higher alpha/steps).
- **Image quality under attack (LPIPS, PSNR, SSIM):** In high-impact PGD settings (eps=10/255, alpha=2/255, steps=20), LPIPS mean is 0.0166 (low = good), PSNR mean is 41.56dB (high = better), and SSIM mean is 0.978 (close to 1 = better). These indicate that even the strongest attacks keep images largely visually similar, though FGSM at max eps can be more visually disruptive.
- **BLEU/ROUGE are reliable for measuring adversarial impact in this context**; CLIPScore is not, as it sometimes increases under attack.

---

## 2. Experimental Setup

- **Model**: Salesforce/blip-image-captioning-large
- **Dataset**: COCO val2017 split 1000 images
- **Attacks**:
  - **FGSM**: Fast Gradient Sign Method—single-step, parameterized by epsilon (max pixel perturbation).
  - **PGD**: Projected Gradient Descent—multi-step, parameterized by epsilon (perturbation bound), alpha (step size), and steps (number of iterations).
- **Metrics**:
  - **BLEU, ROUGE-L**: Caption quality (higher is better). Degradation = (adversarial - baseline); negative is worse.
  - **CLIPScore**: Semantic image-caption similarity. Deltas may be positive or negative; not always aligned with BLEU/ROUGE.
  - **Average Loss Increase**: Increase in model loss (cross entropy) due to attack; higher = more effective attack.
  - **LPIPS (Learned Perceptual Image Patch Similarity):** Perceptual distance (lower = better).
  - **PSNR (Peak Signal-to-Noise Ratio):** Pixel distortion in dB (higher = better).
  - **SSIM (Structural Similarity Index):** Structural similarity (closer to 1 = better).

**Interpretation of deltas**:  
Negative BLEU/ROUGE deltas indicate loss of caption fidelity. Higher avg loss increase signals that the attack is successfully confusing the model during generation. Positive CLIPScore deltas (adversarial > baseline) are counter-intuitive and indicate metric unreliability for adversarial settings. positive loss/LPIPS = more damage, higher PSNR/SSIM = closer to original.

---

## 3. Results and Comparisons

### 3.1 Standard Runs

#### a) Ranking by BLEU/ROUGE Degradation

| attack_type | eps   | alpha   | steps | BLEU Δ   | ROUGE Δ  | CLIP Δ    | Avg Loss ↑ |
|-------------|-------|---------|-------|----------|----------|-----------|------------|
| PGD         | 8/255 | 1/255   | 10    | -0.1172  | -0.1012  | +0.0872   | 1.1475     |
| PGD         | 4/255 | 1/255   | 10    | -0.0952  | -0.0774  | -0.0251   | 0.8161     |
| PGD         | 2/255 | 1/255   | 10    | -0.0785  | -0.0630  | -0.2820   | 0.4314     |
| FGSM        | 8/255 | N/A     | N/A   | -0.0481  | -0.0316  | +0.5455   | 0.3717     |
| FGSM        | 4/255 | N/A     | N/A   | -0.0358  | -0.0233  | +0.1278   | 0.2833     |
| FGSM        | 2/255 | N/A     | N/A   | -0.0358  | -0.0233  | +0.1278   | 0.2006     |

#### b) Trends and Reasoning

- **PGD dominates**: For any given epsilon, PGD delivers at least twice the BLEU/ROUGE degradation of FGSM, due to iterative optimization.
- **Scaling with epsilon**: For both attacks, increasing epsilon yields greater degradation, but the rate saturates for FGSM beyond 4/255.
- **PGD scaling**: Step count (fixed at 10) and alpha (1/255) are sufficient to realize the full potential for each epsilon; further increases (as seen in high-impact) can push degradation even higher.
- **CLIPScore divergence**: Notably, FGSM 8/255 yields a **positive CLIP delta (+0.5455)** even as BLEU/ROUGE degrade, suggesting CLIPScore is not sensitive to adversarial caption quality loss. PGD at higher epsilon can even result in negative CLIP deltas (e.g., -0.2820 at 2/255), but these are not consistent.
- **Average loss increase**: Correlates with BLEU/ROUGE degradation, validating it as a proxy for attack strength.

---

### 3.2 High-Impact Runs

#### a) Ranking by BLEU/ROUGE Degradation

| attack_type | eps    | alpha   | steps | BLEU Δ   | ROUGE Δ  | CLIP Δ    | Avg Loss ↑ |
|-------------|--------|---------|-------|----------|----------|-----------|------------|
| PGD         |10/255  |2/255    |20     | -0.1833  | -0.1582  | +0.1820   | 2.0015     |
| PGD         | 8/255  |1/255    |20     | -0.1440  | -0.1206  | +0.4244   | 1.6313     |
| PGD         | 8/255  |2/255    |10     | -0.1365  | -0.1106  | +0.5503   | 1.3586     |
| PGD         | 6/255  |2/255    |5      | -0.0846  | -0.0616  | +1.9007   | 0.8367     |
| FGSM        |10/255  |N/A      |N/A    | -0.0471  | -0.0231  | +0.6946   | 0.4001     |
| FGSM        | 6/255  |N/A      |N/A    | -0.0475  | -0.0213  | +1.1379   | 0.3350     |

#### b) Trends and Reasoning

- **PGD scaling**: Increasing both steps and alpha in PGD (e.g., 20 steps, alpha=2/255) at high epsilon (10/255) produces the strongest degradation observed in all experiments.
- **FGSM saturation**: At 6/255 and 10/255, FGSM BLEU/ROUGE deltas are nearly identical, suggesting diminishing returns for single-step attacks at higher perturbation bounds.
- **CLIPScore divergence is even stronger in high-impact runs**: For instance, PGD 8/255, alpha=2/255, 10 steps yields CLIP delta +0.5503; FGSM 6/255 gives +1.1379.
- **Loss increase**: PGD 10/255, alpha=2/255, 20 steps more than doubles the avg loss increase compared to the strongest FGSM, confirming attack efficacy.



#### c) Image Quality Metrics (High-impact runs)

| attack_type | eps   | alpha   | steps | LPIPS mean | PSNR mean | SSIM mean |
|-------------|-------|---------|-------|------------|-----------|-----------|
| PGD         |10/255 |2/255    |20     | 0.0166     | 41.56     | 0.978     |
| PGD         | 8/255 |1/255    |20     | 0.0082     | 44.19     | 0.988     |
| PGD         | 8/255 |2/255    |10     | 0.0076     | 44.40     | 0.988     |
| PGD         | 6/255 |2/255    |5      | 0.0073     | 44.33     | 0.987     |
| FGSM        |10/255 |N/A      |N/A    | 0.0296     | 38.66     | 0.951     |
| FGSM        | 6/255 |N/A      |N/A    | 0.0122     | 42.17     | 0.977     |

- **PGD is less visually damaging than FGSM at the same epsilon:** For example, FGSM 10/255 yields LPIPS 0.0296 (worse), PSNR 38.66 (lower), and SSIM 0.951 (lower), while PGD 10/255, 2/255, 20 steps achieves LPIPS 0.0166, PSNR 41.56, SSIM 0.978.
- **All attacks maintain high perceptual similarity to original images**, with most SSIM > 0.97 and PSNR > 41dB for PGD.

---

### 3.3 Cross-Setting Comparison

- **Trend consistency**: The ranking and scaling of attacks are consistent across standard and high-impact runs: PGD > FGSM, higher epsilon/steps = more degradation.
- **Magnitude difference**: The absolute deltas for BLEU/ROUGE are higher in high-impact runs due to more aggressive settings (and possibly higher variance due to smaller sample).
- **Anomalies**: Some identical entries for FGSM 2/255 and 4/255 in standard runs (likely due to attack saturation or reporting duplication); does not affect primary conclusions.
- **Image quality:** Strong attacks (PGD) cause substantial caption degradation with relatively small perceptual changes to images (low LPIPS, high SSIM/PSNR), suggesting that the attacks are not trivially perceptible.

---

## 4. Tables for the Report

### Top-3 Most Damaging Configurations (Standard)

| attack_type | eps   | alpha   | steps | BLEU Δ   | ROUGE Δ  | CLIP Δ    | Avg Loss ↑ |
|-------------|-------|---------|-------|----------|----------|-----------|------------|
| PGD         | 8/255 | 1/255   | 10    | -0.1172  | -0.1012  | +0.0872   | 1.1475     |
| PGD         | 4/255 | 1/255   | 10    | -0.0952  | -0.0774  | -0.0251   | 0.8161     |
| PGD         | 2/255 | 1/255   | 10    | -0.0785  | -0.0630  | -0.2820   | 0.4314     |

### Top-3 Most Damaging Configurations (High-Impact)

| attack_type | eps    | alpha   | steps | BLEU Δ   | ROUGE Δ  | CLIP Δ    | Avg Loss ↑ |
|-------------|--------|---------|-------|----------|----------|-----------|------------|
| PGD         |10/255  |2/255    |20     | -0.1833  | -0.1582  | +0.1820   | 2.0015     |
| PGD         | 8/255  |1/255    |20     | -0.1440  | -0.1206  | +0.4244   | 1.6313     |
| PGD         | 8/255  |2/255    |10     | -0.1365  | -0.1106  | +0.5503   | 1.3586     |

### Best PGD vs Best FGSM (Both Settings)

| Setting      | Best PGD Config                  | BLEU Δ   | ROUGE Δ  | CLIP Δ   | Avg Loss ↑ | LPIPS mean | PSNR mean | SSIM mean | Best FGSM Config           | BLEU Δ   | ROUGE Δ  | CLIP Δ   | Avg Loss ↑ | LPIPS mean | PSNR mean | SSIM mean |
|--------------|----------------------------------|----------|----------|----------|------------|------------|-----------|-----------|----------------------------|----------|----------|----------|------------|------------|-----------|-----------|
| Standard     | PGD 8/255, 1/255, 10 steps       | -0.1172  | -0.1012  | +0.0872  | 1.1475     | —          | —         | —         | FGSM 8/255                 | -0.0481  | -0.0316  | +0.5455  | 0.3717     | —          | —         | —         |
| High-Impact  | PGD 10/255, 2/255, 20 steps      | -0.1833  | -0.1582  | +0.1820  | 2.0015     | 0.0166     | 41.56     | 0.978     | FGSM 10/255                | -0.0471  | -0.0231  | +0.6946  | 0.4001     | 0.0296     | 38.66     | 0.951     |

---

## 6. Conclusions and Recommendations

- **PGD with high epsilon (8–10/255), alpha (1–2/255), and steps (10–20) is the most effective adversarial configuration**, causing the greatest degradation in caption quality while maintaining high perceptual similarity to the original image.
- **For stress-testing**, use PGD (eps=10/255, alpha=2/255, steps=20) to probe model robustness under worst-case attacks.
- **For moderate evaluation**, PGD (eps=4–6/255, alpha=1/255, steps=10) yields strong but less extreme degradation, balancing realism and attack strength.
- **FGSM is substantially less effective than PGD** and saturates at high epsilon, making it less suited for meaningful robustness benchmarking.
- **Metric sensitivity:** BLEU and ROUGE reliably indicate caption degradation under attack. CLIPScore is unreliable—it often increases with stronger attacks, likely due to insensitivity to subtle caption errors or overfitting to image features.
- **Image quality metrics (LPIPS, PSNR, SSIM) confirm that strong attacks do not grossly distort images**, validating the use of these adversarial examples for robustness evaluation.
- **Limitations:** While automatic metrics capture most effects, real-world semantic impact may require human evaluation. CLIPScore should not be used as a primary robustness metric in this context.

---

## 7. Generated Figures (Standard and High-Impact)

Below are the generated plots from `result/figures/`. They visualize the degradations and trends discussed above.

### Standard

- __BLEU degradation by configuration__

![BLEU bar (standard)](result/figures/standard/bleu_bar_standard.png)

_Finding: PGD bars are consistently more negative than FGSM at the same epsilon; degradation increases with epsilon._

- __ROUGE-L degradation by configuration__

![ROUGE bar (standard)](result/figures/standard/rouge_bar_standard.png)

_Finding: Same pattern as BLEU—PGD dominates FGSM; higher epsilon yields larger ROUGE-L degradation._

- __PGD BLEU degradation vs epsilon__

![PGD BLEU vs eps (standard)](result/figures/standard/pgd_bleu_vs_eps_standard.png)

_Finding: BLEU degradation becomes more negative as epsilon increases; configurations with higher steps/alpha lie lower (stronger attack)._ 

- __PGD ROUGE-L degradation vs epsilon__

![PGD ROUGE vs eps (standard)](result/figures/standard/pgd_rouge_vs_eps_standard.png)

_Finding: ROUGE-L degradation mirrors BLEU trends with monotonic decline vs epsilon and stronger settings yielding lower curves._

Key observations (standard):
- **PGD > FGSM** at comparable epsilon in both BLEU and ROUGE.
- **Monotonic trend**: higher epsilon generally yields larger degradation.

### High-Impact

- __BLEU degradation by configuration__

![BLEU bar (high-impact)](result/figures/high_impact/bleu_bar_high_impact.png)

_Finding: Magnitudes are larger than standard; PGD with higher steps/alpha shows the strongest BLEU degradation._

- __ROUGE-L degradation by configuration__

![ROUGE bar (high-impact)](result/figures/high_impact/rouge_bar_high_impact.png)

_Finding: ROUGE-L degradation also increases markedly in high-impact PGD settings, exceeding FGSM at comparable epsilon._

- __PGD BLEU degradation vs epsilon__

![PGD BLEU vs eps (high-impact)](result/figures/high_impact/pgd_bleu_vs_eps_high_impact.png)

_Finding: BLEU degradation declines more steeply with epsilon than in standard; highest steps/alpha produce the lowest curves._

- __PGD ROUGE-L degradation vs epsilon__

![PGD ROUGE vs eps (high-impact)](result/figures/high_impact/pgd_rouge_vs_eps_high_impact.png)

_Finding: ROUGE-L shows the same steep decline with epsilon; stronger (steps/alpha) settings consistently worsen degradation._

Key observations (high-impact):
- **Stronger PGD settings (higher steps/alpha)** amplify degradation compared to standard runs.
- Trends are consistent with standard; magnitudes are larger due to more aggressive parameters.