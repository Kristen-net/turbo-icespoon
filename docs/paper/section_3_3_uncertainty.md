# §3.3 Multi-Task Uncertainty Weighting

## 3.3.1 Problem Formulation

The joint loss in §3.2 combines five heterogeneous terms: pixel-level reconstruction, feature-level preservation, detection confidence, box regression, and physical consistency. These losses operate at vastly different scales and gradient magnitudes — a pixel-level L1 loss produces gradients $\sim O(1)$ in image space, while a CIoU loss produces gradients $\sim O(0.01)$ in geometry space. Manually tuning $K$-dimensional $\lambda$ weights requires exhaustive grid search and is highly sensitive to learning rate.

We adopt the **learnable task uncertainty** framework of Kendall & Gal (CVPR 2018) to automatically balance these losses.

## 3.3.2 Formulation

For $K$ loss terms $\{\mathcal{L}_1, \ldots, \mathcal{L}_K\}$, the total loss is:

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{K} \exp(-s_i) \cdot \mathcal{L}_i + \frac{1}{2} \sum_{i=1}^{K} s_i$$

where $s_i = \log \sigma_i^2$ are learnable parameters. The weight of each loss is $w_i = \exp(-s_i) = 1/\sigma_i^2$. The second term $\frac{1}{2}\sum s_i$ serves as a regularizer, preventing the trivial solution $s_i \to \infty$ (which would zero out all losses).

| $i$ | Loss | Physical Meaning |
|-----|------|------------------|
| 1 | $\mathcal{L}_{\text{recon}}$ | Pixel-level reconstruction (L1 + SSIM) |
| 2 | $\mathcal{L}_{\text{box-feat}}$ | Box-internal feature preservation (C2) |
| 3 | $\mathcal{L}_{\text{detectability}}$ | Detection confidence (C2) |
| 4 | $\mathcal{L}_{\text{box-align}}$ | CIoU + DFL (C2) |
| 5 | $\mathcal{L}_{\text{ice-phys}}$ | Physical consistency (C1, synthetic only) |

## 3.3.3 σ Drift Constraint

In practice, the uncertainty parameters $s_i$ can drift to extreme values during training, especially when one loss dominates early epochs. To prevent any loss from being effectively "zeroed out" ($s_i \to +\infty$) or dominating ($s_i \to -\infty$), we clamp:

$$s_i \in [-6.0, 6.0]$$

This bounds the effective weight to $w_i \in [e^{-6}, e^{6}] \approx [0.0025, 403]$, ensuring all loss terms contribute meaningfully throughout training. We monitor $s_i$ every 5 epochs and log the values to the training checkpoint.

## 3.3.4 Advantages over Manual λ

| Dimension | Manual λ | Learned $s_i$ |
|-----------|----------|---------------|
| Tuning effort | High ($K$-dim grid) | Near-zero (single training run) |
| Scale adaptation | No | Yes (gradient magnitude absorbed) |
| LR sensitivity | High | Low |
| API compatibility | — | Drop-in (`UncertaintyWeighting(K=5)`) |
| Interpretability | Direct | $s_i$ reveals relative task difficulty |

## 3.3.5 Implementation Details

The uncertainty parameters $s_i$ are initialized to 0 (initial weight $w_i = 1.0$), added to the optimizer alongside model parameters, and saved in the checkpoint under `state["uncertainty"]`. During Stage-A (warm-up), only $\mathcal{L}_{\text{recon}}$ and $\mathcal{L}_{\text{ice-phys}}$ are active; the uncertainty module is introduced in Stage-B.

The uncertainty gradients are computed automatically via PyTorch autograd, requiring no special learning rate group or scheduling. The regularizer term $\frac{1}{2}\sum s_i$ provides a natural gradient pulling $s_i$ toward 0, balanced by the gradient from $\exp(-s_i)\mathcal{L}_i$ which pushes $s_i$ up for larger losses.
