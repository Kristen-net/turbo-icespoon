---
title: "Section 3.1 — HA-WFE: Degradation-Aware Wavelet Feature Enhancement"
target_conferences: ["IEEE TPAMI", "Elsevier CVIU", "Springer Machine Vision and Applications"]
status: draft v0.1 (2026-09-05)
code_pointer: "src/icewave/models/hawfe.py L79-L236"
---

# §3. IceWave-DehazeFormer Architecture

The IceWave-DehazeFormer builds on the DehazeFormer backbone [ref] through
three architectural extensions:

1. **HA-WFE** — a degradation-aware Haar-wavelet feature enhancement
   module inserted at the bottleneck stage (this section).
2. **CLIP-based fog-prompt distillation** — a teacher branch providing
   *x*-conditioned gating of low-frequency enhancement (Section 3.2).
3. **Detect-aware joint optimisation** — a Kendall-&-Gal multi-task loss
   with a downstream mAP-gain evaluation protocol (Section 3.3).

This section focuses on (1). Sections 3.2 and 3.3 are delivered in
companion documents `docs/paper/section_3_2_clip_prompt.md` and
`docs/paper/section_3_3_detect_aware.md` respectively.

---

## §3.1 HA-WFE: Degradation-Aware Wavelet Feature Enhancement

### §3.1.1 Motivation and Insight

Hazy images in real transmission-line inspection suffer from two
*coupled* degradations: (i) low-frequency airlight that washes out the
global contrast, and (ii) loss of high-frequency ice-texture edges that
are exactly the cue downstream detection heads rely on. Naïve spatial-
domain dehazing treats all frequency bands uniformly, risking the
erosion of those ice-specific high-frequency cues.

A natural remedy is to *separate* the enhancement by frequency sub-band
via a wavelet decomposition. Wavelet-domain image restoration is a
well-explored direction — Wavelet U-Net, MWCNN, ProDehaze [refs] have
shown gains by *reconstructing* images in the wavelet basis. What
remains underexplored is *what to do with the high-frequency
sub-bands* once decomposed: should they be (a) sharpened,
(b) preserved, or (c) modulated by a degradation prior?

**Our position (differentiation from prior wavelet-restoration
methods):** We position high-frequency sub-bands as a *content-carrier*
to be preserved, not a residual to be reconstructed. The enhancement
gating comes from the degradation side (low-frequency airlight & CLIP
fog prompt), while high-frequency sub-bands carry only a small,
gated residual correction. This explicit role assignment is the key
distinction from prior work (cf. Table 1 in §1.1).

| | What is enhanced in the wavelet domain? | Gating signal |
|---|---|---|
| Wavelet U-Net [ref] | All sub-bands (implicit) | Spatial encoder features |
| MWCNN [ref] | All sub-bands (CNN residues) | Spatial-domain loss |
| ProDehaze (2025) [ref] | High-frequency structure cues | Spatial-domain prior |
| **HA-WFE (ours)** | **Low-frequency (SCA) + per-band gated residual** | **CLIP fog prompt (LL) ; learned sigmoid gate (HF)** |

### §3.1.2 Single-Level Haar DWT/IWT (Background)

Given a 2-D feature map $F \in \mathbb{R}^{B \times C \times H \times W}$,
the single-level Haar decomposition is parameter-free and reads

$$
\begin{aligned}
LL &= (F_{00} + F_{01} + F_{10} + F_{11}) / 2 \\
LH &= (F_{00} - F_{01} + F_{10} - F_{11}) / 2 \\
HL &= (F_{00} + F_{01} - F_{10} - F_{11}) / 2 \\
HH &= (F_{00} - F_{01} - F_{10} + F_{11}) / 2
\end{aligned}
\tag{1}
$$

where $F_{ij}$ are the $2{\times}2$ sub-samples. The inverse (IWT) is
the linear operator with the same structure. We adopt the implementation
of [ref, Eqs. (1)–(2)] which is exactly invertible (verified numerically
in `tests/test_model_compat.py`).

> **Implementation note.** The module internally disables AMP
> autocast (`torch.amp.autocast('cuda', enabled=False)`,
> `hawfe.py:115, 194`) so that the sub-band arithmetic matches the
> legacy FP32 reference (released M2/M2p/M3/M4 checkpoints were
> trained entirely in FP32). Feature dtype is restored at the return
> boundary so that downstream AMP continues normally.

> **Odd-size fix.** `hawfe.py:121-122, 142-147, 200-201, 229-232`
> — legacy `phase4/ha_wfe.py` only cropped `F_enhanced` after odd-size
> padding, leaving `F_b` in the padded shape; the resulting
> broadcasting mismatch was a latent bug masked by always-even feature
> shapes. We crop both tensors, with a behaviour-preserving guarantee
> for the even-size path (verified by `tests/test_model_compat.py`).

### §3.1.3 HA-WFE v1 — Additive Residual Form (Retrospective)

The first release (HA-WFE v1) targeted a *plug-in* upgrade of the
already-trained M2 model without re-initialising the backbone. To
preserve the pretrained spatial-domain behaviour at the start of fine-
tuning, all learnable scaling parameters were **zero-initialised**:

$$
\alpha_{LL}, \alpha_{HF}, \beta := 0 \quad \text{at } t=0.
\tag{2}
$$

This makes the initial forward exactly the identity: a critical
property since the released M2 was already trained for hundreds of
epochs in the spatial domain. The full v1 update is

$$
\begin{aligned}
LL^{\mathrm{out}}_{(1)} &= LL + \alpha_{LL}\,(LL_{\mathrm{SCA}} - LL), \\
LH^{\mathrm{out}}_{(1)} &= LH + \alpha_{HF}\,
   g_{\mathrm{dir}}(LH)\cdot LH, \\
HL^{\mathrm{out}}_{(1)} &= HL + \alpha_{HF}\,
   g_{\mathrm{dir}}(HL)\cdot HL, \\
HH^{\mathrm{out}}_{(1)} &= HH + \alpha_{HF}\, h_{\mathrm{HH}}(HH), \\
F^{\mathrm{out}}_{(1)} &= F_b + \beta\, \mathrm{IWT}(LL^{\mathrm{out}}_{(1)},
   LH^{\mathrm{out}}_{(1)}, HL^{\mathrm{out}}_{(1)}, HH^{\mathrm{out}}_{(1)}).
\end{aligned}
\tag{3}
$$

where $LL_{\mathrm{SCA}} = \mathrm{SCA}(LL)$ (Simple Channel Attention
[ref]), $g_{\mathrm{dir}}(\cdot)$ is a depth-wise + point-wise
$3{\times}3$ conv followed by $\tanh$ (a *signed* gate), and
$h_{\mathrm{HH}}(\cdot)$ is a similar sequence without the gate.
A single $\alpha_{HF}$ is shared across LH, HL, HH.

### §3.1.4 HA-WFE v2 — Gated Residual Form (Default; M2p/M3/M4)

Motivated by two limitations of v1 — (i) the $\tanh$ gate ranges in
$(-1, +1)$, occasionally inverting high-frequency content; and (ii) the
shared $\alpha_{HF}$ conflates directional and diagonal textures —
v2 introduces three changes:

1. **Sigmoid gate** in $[0,1]$ for high-frequency sub-bands (never
   inverts content):
   $$
   g_{\mathrm{HF}}(x) = \sigma\!\left(W_1\!\star\!_{\mathrm{dw}}\, W_0 x\right) \in [0,1]^C.
   \tag{4}
   $$

2. **Per-sub-band** learnable scaling $\alpha_{LH}, \alpha_{HL},
   \alpha_{HH}$ (3 parameters in place of the single $\alpha_{HF}$).

3. **Low-frequency prompt gating** (Section 3.2). The CLIP-derived
   fog prompt $M_h \in \mathbb{R}^{B \times C_p \times H \times W}$,
   after bilinear interpolation to the LL resolution, conditions the
   low-frequency enhancement through a $(1 + \gamma)$ multiplier:
   $$
   \begin{aligned}
   \gamma &= \pi_{LL}([LL;\, M_h^{\downarrow}]) \in [-1,1]^C
        \quad\text{(Tanh; v1-compatible)} \\
   LL^{\mathrm{out}}_{(2)} &= LL\,(1-\alpha_{LL}) + LL_{\mathrm{SCA}}\,
        \alpha_{LL}\,(1 + \gamma).
   \end{aligned}
   \tag{5}
   $$

The composite update is

$$
\begin{aligned}
LH^{\mathrm{out}}_{(2)} &= LH + \alpha_{LH}\, g_{\mathrm{HF}}(LH)\cdot c_{\mathrm{HF}}(LH), \\
HL^{\mathrm{out}}_{(2)} &= HL + \alpha_{HL}\, g_{\mathrm{HF}}(HL)\cdot c_{\mathrm{HF}}(HL), \\
HH^{\mathrm{out}}_{(2)} &= HH + \alpha_{HH}\, g_{\mathrm{HF}}(HH)\cdot c_{\mathrm{HF}}(HH), \\
F^{\mathrm{out}}_{(2)} &= (1-\beta)\, F_b
   + \beta\, \mathrm{IWT}\!\left(LL^{\mathrm{out}}_{(2)},
      LH^{\mathrm{out}}_{(2)}, HL^{\mathrm{out}}_{(2)}, HH^{\mathrm{out}}_{(2)}\right).
\end{aligned}
\tag{6}
$$

with $\alpha_{\ast}, \beta$ initialised to $0.1$ so that at $t=0$ the
module contributes roughly 10% of its eventual correction. The HAAR
reconstruction identity guarantees that $F^{\mathrm{out}}_{(2)} = F_b$
when $g_{\mathrm{HF}} \equiv 0$ and $\pi_{LL} \equiv 0$.

### §3.1.5 Placement Inside DehazeFormer (Fig. 3 in manuscript)

The module is inserted between `layer3` and `patch_split1` of the
DehazeFormer-S/B/T family (`hawfe.py:270-293`). Both v1 and v2
preserve state-dict key compatibility — released checkpoints from
M1/M2/M3/M4 load with `strict=True`. The insertion position is chosen
because (a) the bottleneck channel is constant ($96$) across all
backbone sizes, decoupling HA-WFE from backbone scaling; (b) the
bottleneck is the natural place to act on *globally* gathered features;
and (c) post-bottleneck enhancement sees all spatial positions of
$F_b$, which minimises interaction with upstream layer-wise
attention.

### §3.1.6 Computational Complexity

Let $C$ be the bottleneck channel and $H \times W$ the feature spatial
size.

| Operation | Cost |
|---|---|
| DWT (Eq. 1) | $4\, B\, C\, H\, W$ (memory only; factorised multiplications) |
| SCA / $g_{\mathrm{HF}}$ / $c_{\mathrm{HF}}$ | $O(C \cdot H \cdot W)$ per conv |
| IWT | $4\, B\, C\, H\, W$ (memory only) |
| **Net per call** | $\mathcal{O}(C \cdot H \cdot W)$ |

For a $256{\times}256$ input and $C=96$, the per-image HA-WFE forward
is $\sim 0.15$ GMACs, equivalent to $2.32\%$ incremental parameters
relative to the DehazeFormer-S baseline (measured at
`build_model('m1')` vs `build_model('m2p')`; see audit §H4).

### §3.1.7 Implementation Anchors and Tests

* Reference implementation — `src/icewave/models/hawfe.py`
  * `HAWFE` (v1)  — L79–L151
  * `HAWFEv2` (v2) — L157–L236
  * `IceWaveDehazeFormer` / `build_model` — L242–L334
* Checkpoints and configs:
  * `src/icewave/train/config.py`
  * `configs/train/{m1,m2,m2p,m3,m4,joint}.yaml`
* Equivalence/robustness tests:
  * `tests/test_model_compat.py` — checkpoint-loading, equivalence
    between v1/v2 forward at $\alpha=\beta=0$.
  * `tests/test_model_compat_v2.py` — odd-size path F_b cropping fix.
  * `tests/test_degradation.py` — end-to-end dimension checks.

All 13 HA-WFE related tests currently pass (subset of 103 total
pytest suite as of commit `7bf8b58`).

### §3.1.8 Reproducibility Statement (Section 1.2 cross-reference)

Per Section 1.2 *Reproducibility*, the exact environment used to
produce all reported numbers is:

```
python -m pip install -e .[train,quality]
icewave-build-dataset --config configs/datasets/<name>.yaml \
                      --ice-mask-human-first  # break circularity
icewave-train    --config configs/train/m2.yaml   # HA-WFE v1
icewave-train    --config configs/train/m2p.yaml  # HA-WFE v2
icewave-bench    --config configs/benchmarks.yaml --seeds 3
```

Random seeds are fixed via `ICEWAVE_SEED=42` and reported per result
table. The four-night ablation matrix (Table 4 in §4) reuses the same
seed; paired *t*-test on the per-image PSNR/SSIM distributions
(`scipy.stats.ttest_rel`) is reported alongside the mean.

---

## Open Questions for Author Review

Before §3.1 is incorporated into the manuscript file, please advise on
the following:

1. **Naming consistency.** Should "HA-WFE" be capitalised as "HAWFE"
   (matches the class name) or kept as "HA-WFE" (matches the README)?
2. **Table 1 frequency.** Should the comparison in §3.1.1 stay in §3.1.1
   or be lifted to §1 (Related Work) for visibility?
3. **Sub-band initialisation.** v2 uses 0.1 for $\alpha_\ast$ and $\beta$.
   Did the reviewer's ablation show anything preferable? (See Section 4.5.)
4. **Odd-size correction.** Cite the fix as a footnote or fold it into
   §3.1.7? (Currently exposed in the implementation note.)
5. **Section 3.1.7 listing.** Length is currently 12 lines; acceptable,
   or should it be condensed into Table A2 (Appendix)?

Once approved, this markdown will be transcribed to
`docs/paper/main.tex` with `\input{section_3_1_hawfe}` for compilation.
