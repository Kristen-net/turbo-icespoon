"""
IceWave-DehazeFormer Phase 2 (M2) - HA-WFE消融实验
模型: DehazeFormer-S + HA-WFE (瓶颈层小波频域增强)
消融对比:
  B0: 原始DehazeFormer-S (= M1基线)
  B1: +HA-WFE (瓶颈层, 差异化子带处理) ← 本脚本
  B2: +HA-WFE (瓶颈层, 统一子带处理)  ← 可选
  B3: +HA-WFE (编码器中层)            ← 可选
配置与M1完全一致 (公平对比)
"""

import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.dehazeformer import dehazeformer_s
from ha_wfe import integrate_hawfe


class Config:
    # 数据
    train_hazy_dir = r"D:\DATA_ALL\dataset\train\hazy"
    train_clear_dir = r"D:\DATA_ALL\dataset\train\clear"
    val_hazy_dir = r"D:\DATA_ALL\dataset\val\hazy"
    val_clear_dir = r"D:\DATA_ALL\dataset\val\clear"

    # 训练 (与M1完全一致)
    epochs = 100
    batch_size = 8
    patch_size = 192
    lr = 2e-4
    weight_decay = 1e-4
    num_workers = 0

    # 损失权重
    lambda_l1 = 1.0
    lambda_ssim = 0.1

    # 输出
    output_dir = r"D:\dehaze_fusion\icewave_output\m2_hawfe"
    save_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    sample_dir = os.path.join(output_dir, "samples")

    # 设备
    device = "cuda"
    seed = 42

config = Config()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, patch_size=192, is_train=True):
        self.hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])
        self.clear_files = sorted([f for f in os.listdir(clear_dir) if f.endswith('.png')])
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.patch_size = patch_size
        self.is_train = is_train

        self.pairs = []
        for hazy_name in self.hazy_files:
            clear_name = '_'.join(hazy_name.replace('.png', '').split('_')[:2]) + '.png'
            if clear_name in self.clear_files:
                self.pairs.append((hazy_name, clear_name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_name, clear_name = self.pairs[idx]
        hazy = cv2.imread(os.path.join(self.hazy_dir, hazy_name))
        clear = cv2.imread(os.path.join(self.clear_dir, clear_name))

        if hazy is None or clear is None:
            return self.__getitem__((idx + 1) % len(self.pairs))

        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)

        if self.is_train:
            h, w = hazy.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                hazy = cv2.resize(hazy, (max(w, self.patch_size), max(h, self.patch_size)))
                clear = cv2.resize(clear, (max(w, self.patch_size), max(h, self.patch_size)))

            h, w = hazy.shape[:2]
            ph, pw = self.patch_size, self.patch_size
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            hazy = hazy[top:top+ph, left:left+pw]
            clear = clear[top:top+ph, left:left+pw]

            if random.random() > 0.5:
                hazy = np.fliplr(hazy).copy()
                clear = np.fliplr(clear).copy()
            if random.random() > 0.5:
                hazy = np.flipud(hazy).copy()
                clear = np.flipud(clear).copy()
        else:
            h, w = hazy.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                hazy = cv2.resize(hazy, (int(w * scale), int(h * scale)))
                clear = cv2.resize(clear, (int(w * scale), int(h * scale)))

        hazy = torch.from_numpy(hazy.astype(np.float32) / 255.0).permute(2, 0, 1)
        clear = torch.from_numpy(clear.astype(np.float32) / 255.0).permute(2, 0, 1)
        return hazy, clear


def validate(model, loader, device):
    psnrs, ssims = [], []
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    with torch.no_grad():
        for hazy, clear in loader:
            hazy = hazy.to(device)
            clear = clear.to(device)
            with torch.amp.autocast(device, dtype=torch.float16):
                pred = model(hazy)
            mse = F.mse_loss(pred, clear)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnrs.append(psnr.item())
            ssim_val = ssim_metric(pred, clear)
            ssims.append(ssim_val.item())
    return np.mean(psnrs), np.mean(ssims)


def save_samples(model, loader, device, save_dir, epoch):
    model.eval()
    with torch.no_grad():
        hazy, clear = next(iter(loader))
        hazy = hazy.to(device)
        with torch.amp.autocast(device, dtype=torch.float16):
            pred = model(hazy)
        hazy_img = hazy[0].cpu().numpy().transpose(1, 2, 0)
        pred_img = pred[0].float().cpu().numpy().transpose(1, 2, 0)
        clear_img = clear[0].numpy().transpose(1, 2, 0)
        combined = np.concatenate([hazy_img, pred_img, clear_img], axis=1)
        combined = np.clip(combined * 255, 0, 255).astype(np.uint8)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"epoch{epoch}.png"), combined)


def train():
    setup_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 2 (M2): DehazeFormer-S + HA-WFE")
    print("=" * 60)

    # 数据
    train_ds = DehazeDataset(config.train_hazy_dir, config.train_clear_dir,
                             config.patch_size, is_train=True)
    val_ds = DehazeDataset(config.val_hazy_dir, config.val_clear_dir, is_train=False)
    print(f"训练: {len(train_ds)}对, 验证: {len(val_ds)}对")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 模型: DehazeFormer-S + HA-WFE
    print("加载模型: DehazeFormer-S + HA-WFE")
    model = dehazeformer_s()
    model = model.to(config.device)
    base_params = sum(p.numel() for p in model.parameters())

    model = integrate_hawfe(model, channels=96)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  基线: {base_params/1e6:.2f}M → +HA-WFE: {new_params/1e6:.2f}M "
          f"(+{(new_params-base_params)/1e3:.1f}K, +{(new_params-base_params)/base_params*100:.1f}%)")

    # 打印HA-WFE的alpha/beta值 (初始应为0)
    print(f"  HA-WFE初始值: alpha_ll={model.hawfe.alpha_ll.item():.4f}, "
          f"alpha_hf={model.hawfe.alpha_hf.item():.4f}, beta={model.hawfe.beta.item():.4f}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    scaler = torch.amp.GradScaler(config.device)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.log_dir)

    best_psnr = 0.0
    print(f"\n开始训练: {config.epochs} epochs, batch={config.batch_size}, patch={config.patch_size}")
    print("=" * 60)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for i, (hazy, clear) in enumerate(train_loader):
            hazy = hazy.to(config.device, non_blocking=True)
            clear = clear.to(config.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(config.device, dtype=torch.float16):
                pred = model(hazy)
                l1 = F.l1_loss(pred, clear)
                ssim_val = 1.0 - ssim_loss(pred, clear)
                loss = config.lambda_l1 * l1 + config.lambda_ssim * ssim_val

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{config.epochs} [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (L1: {l1.item():.4f}, SSIM: {ssim_val.item():.4f}) "
                      f"[{time.time()-t0:.0f}s]", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # 验证
        model.eval()
        val_psnr, val_ssim = validate(model, val_loader, config.device)

        # HA-WFE参数变化
        a_ll = model.hawfe.alpha_ll.item()
        a_hf = model.hawfe.alpha_hf.item()
        beta = model.hawfe.beta.item()

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/psnr", val_psnr, epoch)
        writer.add_scalar("val/ssim", val_ssim, epoch)
        writer.add_scalar("hawfe/alpha_ll", a_ll, epoch)
        writer.add_scalar("hawfe/alpha_hf", a_hf, epoch)
        writer.add_scalar("hawfe/beta", beta, epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{config.epochs} avg_loss={avg_loss:.4f} "
              f"val_PSNR={val_psnr:.2f} val_SSIM={val_ssim:.4f} "
              f"[a_ll={a_ll:.4f} a_hf={a_hf:.4f} b={beta:.4f}] [{elapsed:.0f}s]", flush=True)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "psnr": val_psnr,
                "ssim": val_ssim,
            }, os.path.join(config.save_dir, "m2_best.pth"))
            print(f"  → 保存最佳模型 PSNR={val_psnr:.2f}", flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "psnr": val_psnr,
                "ssim": val_ssim,
            }, os.path.join(config.save_dir, f"m2_epoch{epoch+1}.pth"))

        if (epoch + 1) % 5 == 0:
            save_samples(model, val_loader, config.device, config.sample_dir, epoch + 1)

    writer.close()
    print(f"\n训练完成! 最佳PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    train()
