# §3.2 Detection-Aware Joint Optimization Framework

## 3.2.1 Motivation

Conventional dehazing-detection pipelines follow a cascade paradigm: a dehazing network is first trained to maximize image fidelity (PSNR/SSIM), and a separate detector is then applied to the dehazed output. This decoupled design suffers from a fundamental misalignment — the dehazing objective optimizes pixel-level reconstruction without any awareness of whether the dehazed result actually benefits downstream detection. A dehazer that achieves high PSNR may still over-smooth critical structural cues (edges of insulators, texture of ice accretions on conductors) that the detector relies upon, leading to a paradox where "better-looking" images yield worse detection performance.

We propose a **detection-aware joint optimization** framework that replaces the cascade with an end-to-end trainable system. The core idea: propagate the detection task gradient back to the dehazing backbone through differentiable surrogates, so that the dehazer learns to produce images that are not only visually clean but also maximally detectable.

## 3.2.2 Composite Degradation Model

We model the image formation process of foggy, ice-covered power-line scenes as a four-stage physical cascade:

$$I = \mathcal{K} \star \Big[ \big(J \odot T_0(D) + \alpha_{\text{ice}}(D) \odot S\big) \odot t(x) + A \odot (1 - t(x)) \Big]$$

where:
- $T_0(D) = \exp(-\beta D)$: ice-layer transmittance (Beer-Lambert law)
- $\alpha_{\text{ice}}(D) = 1 - \exp(-\beta D) \in [0,1)$: ice-layer opacity
- $S$: specular reflection on ice surface, decomposed as $S = R_{\text{spec}} \cdot I_{\text{env}} + R_{\text{trans}} \cdot J_{\text{trans}}$, capturing directional specular reflection and isotropic sub-surface scattering, with weights sampled according to ice crystal morphology (shell, sleeve, lobe, cluster)
- $\mathcal{K}$: anisotropic blur kernel ($5 \times 5$, $\sigma=[1.0, 2.5]$, elongated along the conductor direction) simulating ice-induced edge diffusion
- $t(x)$: atmospheric transmittance; $A$: atmospheric light

This model is **fully differentiable** when implemented in PyTorch (convolution with fixed weights), enabling gradient backpropagation from the detection loss through the degradation model to the dehazing network.

## 3.2.3 Detection-Aware Loss Design

The joint loss combines four detection-aware sub-items:

$$\mathcal{L}_{\text{det-aware}} = \mathcal{L}_{\text{box-feat}} + \mathcal{L}_{\text{detectability}} + \mathcal{L}_{\text{box-align}} + \mathcal{L}_{\text{corridor}}$$

### Box-Internal Feature Preservation ($\mathcal{L}_{\text{box-feat}}$)

For each GT bounding box $b_i$ and detection-neck scale $l \in \{P3, P4, P5\}$:

$$\mathcal{L}_{\text{box-feat}} = \sum_{l} \sum_{i=1}^{N} \frac{1}{|b_i|} \sum_{x \in b_i} \big[1 - \cos\big(\Phi_l(R_\theta(I); x), \Phi_l(J; x)\big)\big]$$

This loss anchors the dehazed feature distribution to the haze-free oracle distribution within GT box regions. Cosine similarity preserves **direction** rather than magnitude, preventing reconstruction noise from corrupting feature semantics. The multi-scale design captures complementary information: P3 for small targets (ice accretions, insulator sheds), P5 for large targets (towers).

**Key design decision**: This loss directly replaces the rule-based ITL pseudo-label constraint, **breaking the circular reasoning** of "rules generate labels → rules evaluate performance." The GT boxes come from human annotation, providing an independent evaluation anchor.

### Detectability Loss ($\mathcal{L}_{\text{detectability}}$)

$$\mathcal{L}_{\text{detectability}} = -\frac{1}{N} \sum_{n=1}^{N} \hat{p}_{\text{obj}}^{(n)} \cdot \mathbb{1}\big[\hat{p}_{\text{cls}^{(n)}}^* > \tau_{\text{cls}}\big]$$

This loss maximizes the detector's objectness confidence on the dehazed output, explicitly preventing the dehazer from over-smoothing to the point where "the detector sees nothing." The indicator function restricts the loss to anchors where the GT class probability exceeds $\tau_{\text{cls}} = 0.3$, ignoring background anchors.

### Box-Alignment Loss ($\mathcal{L}_{\text{box-align}}$)

$$\mathcal{L}_{\text{box-align}} = \frac{1}{N} \sum_{n=1}^{N} \big[\lambda_{\text{CIoU}} (1 - \text{CIoU}(\hat{b}_n, b_n^*)) + \lambda_{\text{DFL}} \text{DFL}(\hat{c}_n, c_n^*)\big]$$

This loss aligns the geometric detection results with GT boxes. CIoU considers center distance and aspect ratio; DFL (Distribution Focal Loss) serves as a regularization term. The gradient propagates through the detector head back to the dehazing backbone.

### Ice Physical Consistency Loss ($\mathcal{L}_{\text{ice-phys}}$, optional)

$$\mathcal{L}_{\text{ice-phys}} = \lambda_{\text{trans}} \| \hat{T}(R_\theta(I)) - T^*(D) \|_1 + \lambda_{\alpha} \| 1 - \hat{\alpha}(R_\theta(I)) - \alpha_{\text{ice}}(D) \|_1$$

Two lightweight auxiliary heads (1×1 conv → sigmoid) estimate physical quantities. This loss is only supervised on synthetic data (no $D$ ground truth for real fog), serving as a regularizer for synthetic-to-real transfer.

## 3.2.4 Two-Stage Training Strategy

| Stage | Loss | Frozen | Epochs | Purpose |
|-------|------|--------|--------|---------|
| Stage-A (warm-up) | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{ice-phys}}$ | Detector (full) | 5-10 | Stabilize reconstruction |
| Stage-B (joint) | All 5 losses + uncertainty | Detector backbone | 30-50 | End-to-end optimization |

**Critical**: Stage-B freezes only the detector **backbone**, fine-tuning neck and head. This prevents the detector from drifting toward the dehazed distribution alone — violating the **out-of-distribution generalization** assumption. Only the dehazing backbone ($R_\theta$) receives gradients from all loss terms.
