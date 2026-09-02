"""配置驱动的训练器 (P1-1): 支持 m1/m2p/m3/m4/joint 五种模式.

相对旧 train_m1~m4.py 的改进:
1. 全部超参与路径进 YAML (旧版硬编码 Config 类, 改超参=改代码);
2. ``joint`` 模式 = M4 + 走廊纹理保持损失 (检测感知) + 不确定性加权,
   实现方案中的联合优化框架;
3. 每次训练落盘 config_snapshot.yaml (含 git commit), 结果写 metrics.json,
   供 eval 汇总, 杜绝手工转录数字不一致;
4. 修复旧版两个训练缺陷:
   (a) prompt 投影层从未进入 optimizer (旧版只优化 model.parameters(),
       CLIPFogPrompt.proj 随机初始化且永不更新) → 新版 train_prompt_proj=true;
   (b) 验证集中心裁剪替代随机裁剪 (可复现)。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from icewave.data.dataset import IceAwareDataset
from icewave.losses.detect import CorridorTextureLoss, UncertaintyWeighting
from icewave.losses.itl import ITLLoss
from icewave.models import build_model
from icewave.models.prompt import CLIPFogPrompt, HazeCLIPTeacher
from icewave.train.config import git_commit
from icewave.utils.paths import OUTPUT_DIR
from icewave.utils.seed import seed_everything, worker_init_fn


class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        seed_everything(cfg.get("seed", 42),
                        deterministic=cfg.get("deterministic", False))

        self.device = cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.version = cfg["model"]["version"]          # m1/m2p/m3/m4/joint
        self.out_dir = Path(cfg.get("output_dir", str(OUTPUT_DIR / "runs" / self.version)))
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_dir = self.out_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._snapshot_config()

    # ------------------------------------------------------------------
    def _snapshot_config(self):
        snap = dict(self.cfg)
        snap["_git_commit"] = git_commit()
        snap["_device"] = self.device
        with open(self.out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)

    def _build_datasets(self):
        dcfg = self.cfg["data"]
        common = dict(
            patch_size=dcfg.get("patch_size", 192),
            return_ice=self.version in ("m4", "joint"),
            corridor_from_ice=self.version == "joint",
        )
        train_ds = IceAwareDataset(dcfg["root"], "train",
                                   is_train=True, **common)
        val_ds = IceAwareDataset(dcfg["root"], "val",
                                 is_train=False, **common)
        return train_ds, val_ds

    def _autocast_dtype(self):
        if not torch.cuda.is_available():
            return None
        # bf16 优先 (数值更稳), Ampere+ 支持
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        tcfg = cfg.get("train", {})
        epochs = tcfg.get("epochs", 30)
        batch_size = tcfg.get("batch_size", 4)
        lr = tcfg.get("lr", 1e-4)
        weight_decay = tcfg.get("weight_decay", 1e-4)
        num_workers = tcfg.get("num_workers", 0)
        grad_clip = tcfg.get("grad_clip", 0.0)

        print(f"=== IceWave 训练 [{self.version}] device={self.device} ===")

        # --- 模型 ---
        model = build_model(self.version,
                            backbone=cfg["model"].get("backbone", "s")).to(self.device)

        init_from = cfg["model"].get("init_checkpoint")
        if init_from:
            ckpt = torch.load(init_from, map_location=self.device,
                              weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)
            print(f"  从检查点初始化: {init_from}")

        # --- CLIP 提示 (m3/m4/joint) ---
        prompt_extractor = None
        prompt_drop = tcfg.get("prompt_drop_prob", 0.5)
        if self.version in ("m3", "m4", "joint"):
            prompt_extractor = CLIPFogPrompt(
                prompt_channels=32, device=self.device).to(self.device)
            prompt_extractor.clip.eval()

        # --- 教师蒸馏 (m3/m4/joint) ---
        teacher = None
        if self.version in ("m3", "m4", "joint"):
            teacher = HazeCLIPTeacher(device=self.device)

        # --- 损失 ---
        lcfg = cfg.get("losses", {})
        itl_loss = None
        corridor_loss = None
        uw = None
        use_itl = self.version in ("m4", "joint")
        use_corridor = self.version == "joint"
        if use_itl:
            itl_loss = ITLLoss(
                lambda_region=lcfg.get("itl_lambda_region", 0.5),
                lambda_boundary=lcfg.get("itl_lambda_boundary", 0.3),
                region_term=lcfg.get("itl_region_term", "ssim"),
            ).to(self.device)
        if use_corridor:
            corridor_loss = CorridorTextureLoss(
                lambda_grad=lcfg.get("corridor_lambda_grad", 1.0)
            ).to(self.device)
            n_losses = 4 if teacher is not None else 3
            uw = UncertaintyWeighting(
                num_losses=lcfg.get("n_losses", n_losses)).to(self.device)

        # --- 数据 ---
        train_ds, val_ds = self._build_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn)
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        print(f"  训练集: {train_ds!r}\n  验证集: {val_ds!r}")

        # --- 优化器: 修复 prompt 投影层不更新问题 ---
        params = list(model.parameters())
        if prompt_extractor is not None and tcfg.get("train_prompt_proj", True):
            params += list(prompt_extractor.proj.parameters())
        if uw is not None:
            params += list(uw.parameters())
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

        amp_dtype = self._autocast_dtype()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

        lambda_l1 = lcfg.get("lambda_l1", 1.0)
        lambda_ssim = lcfg.get("lambda_ssim", 0.1)
        lambda_kd = lcfg.get("lambda_kd", 0.05)
        lambda_itl = lcfg.get("lambda_itl", 0.5)

        best_psnr = -1.0
        metrics_history = []
        log_file = open(self.log_dir / "train_log.txt", "w", encoding="utf-8")

        import random as _random

        for epoch in range(epochs):
            model.train()
            t0 = time.time()
            ep = {"loss": 0.0, "recon": 0.0, "kd": 0.0, "itl": 0.0,
                  "corridor": 0.0, "n": 0, "n_prompt": 0}

            for batch in train_loader:
                hazy, clear = batch[0].to(self.device), batch[1].to(self.device)
                ice = batch[2].to(self.device) if len(batch) > 2 else None
                corridor = batch[3].to(self.device) if len(batch) > 3 else None

                optimizer.zero_grad(set_to_none=True)

                use_prompt = (prompt_extractor is not None
                              and _random.random() >= prompt_drop)

                with torch.autocast(
                        "cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    fog_prompt = None
                    if use_prompt:
                        fog_prompt = prompt_extractor(hazy)
                        ep["n_prompt"] += 1
                    pred = model(hazy, fog_prompt=fog_prompt) if hasattr(
                        model, "hawfe") else model(hazy)

                    pred_f, clear_f = pred.float(), clear.float()
                    l1 = F.l1_loss(pred_f, clear_f)
                    ssim_val = 1.0 - _ssim(pred_f, clear_f)
                    loss_recon = lambda_l1 * l1 + lambda_ssim * ssim_val

                    loss_kd = pred_f.sum() * 0.0
                    if teacher is not None:
                        with torch.no_grad():
                            y_teacher = teacher(hazy)
                        loss_kd = lambda_kd * F.l1_loss(pred_f, y_teacher)

                    loss_itl = pred_f.sum() * 0.0
                    if itl_loss is not None and ice is not None:
                        _, _, loss_itl = itl_loss(pred_f, clear_f, ice)
                        loss_itl = lambda_itl * loss_itl

                    loss_corridor = pred_f.sum() * 0.0
                    if corridor_loss is not None and corridor is not None:
                        loss_corridor = corridor_loss(pred_f, clear_f, corridor)

                    if uw is not None:
                        loss = uw([loss_recon, loss_kd, loss_itl, loss_corridor])
                    else:
                        loss = loss_recon + loss_kd + loss_itl

                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optimizer)
                scaler.update()

                ep["loss"] += loss.item()
                ep["recon"] += loss_recon.item()
                ep["kd"] += loss_kd.item()
                ep["itl"] += float(loss_itl)
                ep["corridor"] += float(loss_corridor)
                ep["n"] += 1

            scheduler.step()

            val_psnr, val_ssim = self.validate(model, val_loader,
                                                prompt_extractor)
            row = {
                "epoch": epoch + 1,
                "loss": ep["loss"] / max(ep["n"], 1),
                "recon": ep["recon"] / max(ep["n"], 1),
                "kd": ep["kd"] / max(ep["n"], 1),
                "itl": ep["itl"] / max(ep["n"], 1),
                "corridor": ep["corridor"] / max(ep["n"], 1),
                "n_prompt": ep["n_prompt"],
                "psnr": val_psnr,
                "ssim": val_ssim,
                "sec": round(time.time() - t0, 1),
            }
            metrics_history.append(row)
            line = (f"Epoch {epoch + 1}/{epochs} loss={row['loss']:.4f} "
                    f"psnr={val_psnr:.2f} ssim={val_ssim:.4f} "
                    f"[{row['sec']}s] prompt={row['n_prompt']}/{ep['n']}")
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                state = {"model": model.state_dict(), "epoch": epoch + 1,
                         "best_psnr": best_psnr, "version": self.version}
                if uw is not None:
                    state["uncertainty"] = uw.state_dict()
                torch.save(state, self.ckpt_dir / f"{self.version}_best.pth")

            with open(self.out_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"best_psnr": best_psnr,
                           "history": metrics_history}, f, indent=2)

        log_file.close()
        print(f"=== 完成: best PSNR={best_psnr:.2f}, 输出 {self.out_dir} ===")
        return best_psnr

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, model, val_loader, prompt_extractor=None):
        model.eval()
        psnrs, ssims = [], []
        for batch in val_loader:
            hazy, clear = batch[0].to(self.device), batch[1].to(self.device)
            fog_prompt = None
            if prompt_extractor is not None:
                fog_prompt = prompt_extractor(hazy)
            pred = model(hazy, fog_prompt=fog_prompt) if hasattr(
                model, "hawfe") else model(hazy)
            pred = pred.float().clamp(0, 1)
            mse = F.mse_loss(pred, clear).item()
            psnrs.append(10 * np.log10(1.0 / (mse + 1e-10)))
            ssims.append(_ssim(pred, clear).item())
        model.train()
        return float(np.mean(psnrs)), float(np.mean(ssims))


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """轻量 SSIM (batch 均值), 训练循环内使用."""
    from icewave.losses.itl import ssim_map

    return ssim_map(x.float(), y.float()).mean()
