# §4.3 Downstream Task Gain Evaluation Protocol

## 4.3.1 Motivation

Traditional dehazing benchmarks report only PSNR/SSIM, which measure image fidelity but not downstream utility. A dehazer that achieves +2 dB PSNR may still produce images where a power-line detector performs *worse* than on the original hazy input — an outcome invisible to image-quality metrics. We introduce the **downstream task gain** protocol to directly answer the question reviewers care most about: *does dehazing actually help detection?*

## 4.3.2 Protocol Definition

A single YOLOv8 detector with **frozen weights** is run on three image sets:

1. **Hazy** ($I$): the degraded input, baseline
2. **Dehazed** ($R_\theta(I)$): the dehazing model's output
3. **Clear** ($J$): the haze-free ground truth, upper bound (oracle)

Three derived quantities are reported:

**Dehaze Gain**:
$$\Delta_{\text{gain}} = \text{mAP}_{\text{dehazed}} - \text{mAP}_{\text{hazy}}$$

**Headroom (Residual Gap)**:
$$\Delta_{\text{gap}} = \text{mAP}_{\text{clear}} - \text{mAP}_{\text{dehazed}}$$

**Normalized Recovery Ratio**:
$$R = \frac{\Delta_{\text{gain}}}{\text{mAP}_{\text{clear}} - \text{mAP}_{\text{hazy}}}, \quad R \in (-\infty, 1]$$

**Physical interpretation**:
- $R = 1$: dehazing fully recovers the detection capability lost to haze
- $0 < R < 1$: dehazing helps but leaves residual gap
- $R = 0$: dehazing has no effect on detection
- $R < 0$: dehazing actively **harms** detection (critical negative result)

## 4.3.3 Statistical Reporting

### 95% Bootstrap Confidence Intervals

We compute bootstrap CIs for $\Delta_{\text{gain}}$ by resampling per-image AP values:
- Resample $N$ images with replacement, $B = 1000$ times
- Compute $\Delta_{\text{gain}}$ for each bootstrap sample
- Report the 2.5th and 97.5th percentiles

### Per-Haze-Level Breakdown

Results are stratified by haze density (thin/medium/dense), providing insight into *where* dehazing helps and where it fails. This is critical for power-line inspection, where dense fog scenarios are the most safety-critical.

### Paired t-test

For ablation comparisons (e.g., joint vs. cascade, with/without $\mathcal{L}_{\text{box-feat}}$), we perform a paired two-sided t-test on per-image mAP values, reporting the t-statistic, p-value, and significance at $\alpha = 0.05$.

## 4.3.4 Expected Results Format

| Method | mAP_hazy | mAP_dehazed | mAP_clear | $\Delta_{\text{gain}}$ | $R$ (%) |
|--------|----------|-------------|-----------|------------------------|---------|
| Cascade (M4→YOLO, uncoupled) | 0.412 | 0.483 | 0.602 | +0.071 | 37.4 |
| **Joint (ours)** | 0.412 | 0.531 | 0.602 | **+0.119** | **62.8** |
| Ablation − $\mathcal{L}_{\text{box-feat}}$ | 0.412 | 0.498 | 0.602 | +0.086 | 45.3 |
| Ablation − uncertainty weighting | 0.412 | 0.514 | 0.602 | +0.102 | 53.7 |

> Three-seed experiments, 95% CI reported. Significance markers: * p<0.05, ** p<0.01.

## 4.3.5 Key Differences from PSNR-Only Evaluation

| Dimension | PSNR-only | $\Delta_{\text{gain}}$ + $R$ |
|-----------|-----------|-------------------------------|
| Evaluates | Image fidelity | Dehazing's value to the **task** |
| SOTA comparability | Direct | Indirect (via gap vs. SOTA) |
| Industrial relevance | No direct evidence | **Strong evidence** (power inspection = detection count) |
| Cost | Low | Medium (two detector runs) |
| Negative result visibility | Hidden | **Explicit** ($R < 0$) |

## 4.3.6 Implementation

The protocol is implemented in `src/icewave/eval/downstream.py`:
- `evaluate_set()`: mAP@0.5 per image set (VOC-style all-point interpolation)
- `downstream_gain()`: orchestrates three-set evaluation
- `compute_gain_stats()`: computes $\Delta_{\text{gain}}$, $\Delta_{\text{gap}}$, $R$ with bootstrap CI
- `per_haze_level_report()`: stratified by haze density metadata
- `paired_t_test()`: statistical significance testing between methods

All functions accept a detector implementing `detect(img_bgr) -> list[dict]`, supporting both YOLOv8 and Mask R-CNN backends.
