"""
IceWave-DehazeFormer Phase 3 (M3) - HazeCLIP雾提示蒸馏

模型: DehazeFormer-S + HA-WFE v2 (prompt) + CLIP雾提示注入

核心创新:
  1. CLIP语义理解 → 雾提示特征M_h → 注入HA-WFE低频分支
  2. HazeCLIP教师(MSBDN)知识蒸馏 → 软目标引导
  3. 训练时使用CLIP+教师, 推理时移除 (零额外推理开销)

损失:
  L_total = L_recon + λ_kd * L_kd
  L_recon = L1(pred, clear) + 0.1 * (1 - SSIM(pred, clear))
  L_kd = L1(pred, y_teacher.detach())

参数预算:
  基线: 1.28M → M3: 1.34M (+4.0%, 在10-15%预算内)
  新增: ll_prompt(21.7K) + clip_proj(139.6K), CLIP/MSBDN推理时移除
"""

import sys
sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

import os, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.dehazeformer import dehazeformer_s
from clip_fog_prompt import (
    CLIPFogPrompt,
    HazeCLIPTeacher,
    integrate_hawfe_v2_with_prompt,
)


class Config:
    train_hazy_dir = r"D:\DATA_ALL\dataset\train\hazy"
    train_clear_dir = r"D:\DATA_ALL\dataset\train\clear"
    val_hazy_dir = r"D:\DATA_ALL\dataset\val\hazy"
    val_clear_dir = r"D:\DATA_ALL\dataset\val\clear"
    epochs = 50
    batch_size = 4
    patch_size = 192
    lr = 5e-5
    weight_decay = 1e-4
    num_workers = 0
    lambda_l1 = 1.0
    lambda_ssim = 0.1
    lambda_kd = 0.1
    prompt_drop_prob = 0.5
    prompt_channels = 32
    m2p_checkpoint = r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\checkpoints\m2p_best.pth"
    output_dir = r"D:\dehaze_fusion\icewave_output\m3_clip_distill"
    save_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    sample_dir = os.path.join(output_dir, "samples")
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
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            hazy = hazy[top:top+self.patch_size, left:left+self.patch_size]
            clear = clear[top:top+self.patch_size, left:left+self.patch_size]
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


def validate(model, loader, device, use_prompt=False, clip_extractor=None):
    psnrs, ssims = [], []
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    with torch.no_grad():
        for hazy, clear in loader:
            hazy = hazy.to(device)
            clear = clear.to(device)
            if use_prompt and clip_extractor is not None:
                model.clip_prompt = clip_extractor(hazy)
            else:
                model.clip_prompt = None
            with torch.amp.autocast(device, dtype=torch.float16):
                pred = model(hazy)
            mse = F.mse_loss(pred, clear)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnrs.append(psnr.item())
            ssims.append(ssim_metric(pred, clear).item())
    return np.mean(psnrs), np.mean(ssims)


def save_samples(model, loader, device, save_dir, epoch, clip_prompt_extractor=None):
    model.eval()
    with torch.no_grad():
        hazy, clear = next(iter(loader))
        hazy = hazy.to(device)
        if clip_prompt_extractor is not None:
            model.clip_prompt = clip_prompt_extractor(hazy)
        else:
            model.clip_prompt = None
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

    print("=" * 70)
    print("Phase 3 (M3): DehazeFormer-S + HA-WFE v2 + CLIP雾提示蒸馏")
    print("  教师: HazeCLIP (MSBDN + CLIPSurgery), 冻结")
    print("  学生: DehazeFormer-S + HA-WFE v2 (prompt_channels=32)")
    print("  蒸馏: CLIP雾提示注入 + 教师输出L1蒸馏")
    print("  推理: 仅学生, 无CLIP/MSBDN, 零额外开销")
    print("=" * 70)

    train_ds = DehazeDataset(config.train_hazy_dir, config.train_clear_dir,
                             config.patch_size, is_train=True)
    val_ds = DehazeDataset(config.val_hazy_dir, config.val_clear_dir, is_train=False)
    print(f"训练: {len(train_ds)}对, 验证: {len(val_ds)}对")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\n加载CLIP雾提示提取器 (CS-ViT-B/32)...")
    clip_prompt_extractor = CLIPFogPrompt(prompt_channels=config.prompt_channels,
                                          device=config.device).to(config.device)
    clip_trainable = sum(p.numel() for p in clip_prompt_extractor.proj.parameters())
    print(f"  CLIP: 151.3M (frozen), 投影层: {clip_trainable/1e3:.1f}K (trainable)")

    print("加载HazeCLIP教师模型 (MSBDN)...")
    teacher = HazeCLIPTeacher(device=config.device).to(config.device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  教师参数: {teacher_params/1e6:.1f}M (frozen)")

    print("\n加载学生模型: DehazeFormer-S + HA-WFE v2 (prompt)...")
    model = dehazeformer_s().to(config.device)
    base_params = sum(p.numel() for p in model.parameters())
    model = integrate_hawfe_v2_with_prompt(model, channels=96,
                                           prompt_channels=config.prompt_channels)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  基线: {base_params/1e6:.2f}M → M3: {new_params/1e6:.2f}M "
          f"(+{(new_params-base_params)/1e3:.1f}K, +{(new_params-base_params)/base_params*100:.1f}%)")

    print(f"\n加载M2'预训练权重: {config.m2p_checkpoint}")
    ckpt = torch.load(config.m2p_checkpoint, map_location=config.device, weights_only=False)
    ckpt_state = ckpt["model"] if "model" in ckpt else ckpt

    own_state = model.state_dict()
    loaded_keys = []
    missing_keys = []
    for k, v in ckpt_state.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v
            loaded_keys.append(k)
        else:
            missing_keys.append(k)
    model.load_state_dict(own_state, strict=True)
    print(f"  已加载: {len(loaded_keys)} keys, 跳过: {len(missing_keys)} keys (新参数)")
    if missing_keys:
        new_keys = [k for k in model.state_dict().keys() if k not in ckpt_state]
        print(f"  新参数: {new_keys}")

    trainable_params = list(model.parameters()) + list(clip_prompt_extractor.proj.parameters())
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    scaler = torch.amp.GradScaler(config.device)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.log_dir)

    best_psnr = 0.0
    print(f"\n开始训练: {config.epochs} epochs, batch={config.batch_size}, patch={config.patch_size}")
    print(f"  λ_l1={config.lambda_l1}, λ_ssim={config.lambda_ssim}, λ_kd={config.lambda_kd}")
    print(f"  prompt_drop_prob={config.prompt_drop_prob} (50%批次无CLIP提示, 保证推理时无CLIP也能工作)")
    print("=" * 70)

    for epoch in range(config.epochs):
        model.train()
        clip_prompt_extractor.proj.train()
        teacher.eval()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kd = 0
        n_batches = 0
        n_with_prompt = 0
        t0 = time.time()

        for i, (hazy, clear) in enumerate(train_loader):
            hazy = hazy.to(config.device, non_blocking=True)
            clear = clear.to(config.device, non_blocking=True)
            optimizer.zero_grad()

            use_prompt = random.random() >= config.prompt_drop_prob

            with torch.amp.autocast(config.device, dtype=torch.float16):
                if use_prompt:
                    M_h = clip_prompt_extractor(hazy)
                    model.clip_prompt = M_h
                    n_with_prompt += 1
                else:
                    model.clip_prompt = None

                pred = model(hazy)

                l1 = F.l1_loss(pred, clear)
                ssim_val = 1.0 - ssim_loss(pred, clear)
                loss_recon = config.lambda_l1 * l1 + config.lambda_ssim * ssim_val

                with torch.no_grad():
                    y_teacher = teacher(hazy)
                loss_kd = config.lambda_kd * F.l1_loss(pred, y_teacher)

                loss = loss_recon + loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_kd += loss_kd.item()
            n_batches += 1

            if (i + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{config.epochs} [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (recon={loss_recon.item():.4f}, "
                      f"kd={loss_kd.item():.4f}) [{'P' if use_prompt else 'N'}] "
                      f"[{time.time()-t0:.0f}s]", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kd = epoch_kd / n_batches

        model.eval()
        clip_prompt_extractor.proj.eval()
        val_psnr, val_ssim = validate(model, val_loader, config.device,
                                     use_prompt=False, clip_extractor=None)
        val_psnr_p, val_ssim_p = validate(model, val_loader, config.device,
                                          use_prompt=True, clip_extractor=clip_prompt_extractor)

        fog_p, clear_p = clip_prompt_extractor.get_fog_confidence(
            next(iter(val_loader))[0].to(config.device))

        a_ll = model.hawfe.alpha_ll.item()
        a_lh = model.hawfe.alpha_lh.item()
        a_hl = model.hawfe.alpha_hl.item()
        a_hh = model.hawfe.alpha_hh.item()
        beta = model.hawfe.beta.item()

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/recon_loss", avg_recon, epoch)
        writer.add_scalar("train/kd_loss", avg_kd, epoch)
        writer.add_scalar("val/psnr_no_prompt", val_psnr, epoch)
        writer.add_scalar("val/ssim_no_prompt", val_ssim, epoch)
        writer.add_scalar("val/psnr_with_prompt", val_psnr_p, epoch)
        writer.add_scalar("val/ssim_with_prompt", val_ssim_p, epoch)
        writer.add_scalar("hawfe/alpha_ll", a_ll, epoch)
        writer.add_scalar("hawfe/alpha_lh", a_lh, epoch)
        writer.add_scalar("hawfe/alpha_hl", a_hl, epoch)
        writer.add_scalar("hawfe/alpha_hh", a_hh, epoch)
        writer.add_scalar("hawfe/beta", beta, epoch)
        writer.add_scalar("clip/fog_confidence", fog_p, epoch)
        writer.add_scalar("clip/clear_confidence", clear_p, epoch)

        elapsed = time.time() - t0
        delta = val_psnr_p - val_psnr
        print(f"Epoch {epoch+1}/{config.epochs} loss={avg_loss:.4f} "
              f"(recon={avg_recon:.4f}, kd={avg_kd:.4f}) "
              f"PSNR={val_psnr:.2f}(nop) {val_psnr_p:.2f}(p) [{delta:+.2f}] "
              f"SSIM={val_ssim:.4f} "
              f"[fog={fog_p:.3f}] [p={n_with_prompt}/{n_batches}] "
              f"[{elapsed:.0f}s]", flush=True)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "clip_proj": clip_prompt_extractor.proj.state_dict(),
                "psnr": val_psnr,
                "ssim": val_ssim,
                "psnr_prompt": val_psnr_p,
                "ssim_prompt": val_ssim_p,
            }, os.path.join(config.save_dir, "m3_best.pth"))
            print(f"  -> 保存最佳 PSNR={val_psnr:.2f} (nop), {val_psnr_p:.2f} (p)", flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "clip_proj": clip_prompt_extractor.proj.state_dict(),
                "psnr": val_psnr,
                "ssim": val_ssim,
            }, os.path.join(config.save_dir, f"m3_epoch{epoch+1}.pth"))

        if (epoch + 1) % 5 == 0:
            save_samples(model, val_loader, config.device, config.sample_dir, epoch + 1,
                        clip_prompt_extractor)

        if config.device == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    print(f"\nPhase 3 训练完成! 最佳PSNR: {best_psnr:.2f}")
    print(f"  M1基线: 34.79, M2': 35.01, M3: {best_psnr:.2f}")
    print(f"  检查点: {os.path.join(config.save_dir, 'm3_best.pth')}")


if __name__ == "__main__":
    train()
