"""
IceWave-DehazeFormer Phase 4 (M4) - 覆冰感知ITL损失

模型: DehazeFormer-S + HA-WFE v2 (prompt) + CLIP蒸馏 + ITL损失

核心创新:
  1. 在M3基础上增加ITL (Ice-aware Territory Loss)
  2. ITL区域约束: 冰区内加权重建, 防止过度平滑
  3. ITL边界约束: 冰/非冰边界梯度保持, 维持覆冰边缘锐利
  4. 解决通用去雾方法过度平滑白色覆冰区域的问题

损失:
  L_total = L_recon + λ_kd * L_kd + λ_itl * L_itl
  L_recon = L1(pred, clear) + 0.1 * (1 - SSIM(pred, clear))
  L_kd = L1(pred, y_teacher.detach())
  L_itl = 0.5 * L_region + 0.3 * L_boundary

训练策略:
  - 从M3最佳检查点初始化 (不是从头训练)
  - 较低学习率 (1e-5) 微调
  - 保持50% prompt dropout
  - 30 epochs (ITL是约束增强, 不需要长训练)
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
from itl_loss import ITLLoss


class Config:
    train_hazy_dir = r"D:\DATA_ALL\dataset\train\hazy"
    train_clear_dir = r"D:\DATA_ALL\dataset\train\clear"
    train_ice_dir = r"D:\DATA_ALL\dataset\train\ice_mask"
    val_hazy_dir = r"D:\DATA_ALL\dataset\val\hazy"
    val_clear_dir = r"D:\DATA_ALL\dataset\val\clear"
    val_ice_dir = r"D:\DATA_ALL\dataset\val\ice_mask"

    epochs = 30
    batch_size = 4
    patch_size = 192
    lr = 1e-5
    weight_decay = 1e-4
    num_workers = 0

    lambda_l1 = 1.0
    lambda_ssim = 0.1
    lambda_kd = 0.05
    lambda_itl = 0.5
    prompt_drop_prob = 0.5
    prompt_channels = 32

    m3_checkpoint = r"D:\dehaze_fusion\icewave_output\m3_clip_distill\checkpoints\m3_best.pth"
    output_dir = r"D:\dehaze_fusion\icewave_output\m4_itl"
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


class IceAwareDataset(Dataset):
    """带覆冰掩码的去雾数据集"""

    def __init__(self, hazy_dir, clear_dir, ice_dir, patch_size=192, is_train=True):
        self.hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.png')])
        self.clear_files = set(os.listdir(clear_dir))
        self.ice_files = set(os.listdir(ice_dir)) if os.path.isdir(ice_dir) else set()
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.ice_dir = ice_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.pairs = []

        for hazy_name in self.hazy_files:
            base = '_'.join(hazy_name.replace('.png', '').split('_')[:2])
            clear_name = base + '.png'
            ice_name = base + '_ice.png'
            if clear_name in self.clear_files:
                has_ice = ice_name in self.ice_files
                self.pairs.append((hazy_name, clear_name, ice_name, has_ice))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_name, clear_name, ice_name, has_ice = self.pairs[idx]
        hazy = cv2.imread(os.path.join(self.hazy_dir, hazy_name))
        clear = cv2.imread(os.path.join(self.clear_dir, clear_name))
        if hazy is None or clear is None:
            return self.__getitem__((idx + 1) % len(self.pairs))

        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)

        # 加载覆冰掩码
        if has_ice and os.path.isdir(self.ice_dir):
            ice = cv2.imread(os.path.join(self.ice_dir, ice_name), cv2.IMREAD_GRAYSCALE)
            if ice is None:
                ice = np.zeros(hazy.shape[:2], dtype=np.uint8)
        else:
            ice = np.zeros(hazy.shape[:2], dtype=np.uint8)

        if self.is_train:
            h, w = hazy.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                new_h = max(h, self.patch_size)
                new_w = max(w, self.patch_size)
                hazy = cv2.resize(hazy, (new_w, new_h))
                clear = cv2.resize(clear, (new_w, new_h))
                ice = cv2.resize(ice, (new_w, new_h))
            h, w = hazy.shape[:2]
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            hazy = hazy[top:top+self.patch_size, left:left+self.patch_size]
            clear = clear[top:top+self.patch_size, left:left+self.patch_size]
            ice = ice[top:top+self.patch_size, left:left+self.patch_size]
        else:
            h, w = hazy.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                new_h = max(h, self.patch_size)
                new_w = max(w, self.patch_size)
                hazy = cv2.resize(hazy, (new_w, new_h))
                clear = cv2.resize(clear, (new_w, new_h))
                ice = cv2.resize(ice, (new_w, new_h))

        hazy = torch.from_numpy(hazy.transpose(2, 0, 1)).float() / 255.0
        clear = torch.from_numpy(clear.transpose(2, 0, 1)).float() / 255.0
        ice = torch.from_numpy(ice[None]).float() / 255.0

        return hazy, clear, ice


def validate(model, val_loader, device, use_prompt=False, clip_extractor=None):
    model.eval()
    psnrs, ssims = [], []
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy = hazy.to(device)
            clear = clear.to(device)
            if use_prompt and clip_extractor is not None:
                M_h = clip_extractor(hazy)
                model.clip_prompt = M_h
            else:
                model.clip_prompt = None

            with torch.amp.autocast(device, dtype=torch.float16):
                pred = model(hazy).float()

            pred = pred.clamp(0, 1)
            mse = F.mse_loss(pred, clear).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            psnrs.append(psnr)
            ssim_metric(pred, clear)
            ssims.append(ssim_metric.compute().item())
            ssim_metric.reset()

    return np.mean(psnrs), np.mean(ssims)


def train():
    setup_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 4 (M4): 覆冰感知ITL损失训练")
    print("=" * 70)

    # --- 1. 模型加载 (从M3初始化) ---
    print("\n[1] 加载M3检查点...")
    model = dehazeformer_s().to(config.device)
    model = integrate_hawfe_v2_with_prompt(
        model, channels=96, prompt_channels=config.prompt_channels
    )

    ckpt = torch.load(config.m3_checkpoint, map_location=config.device,
                      weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"  M3检查点已加载: {config.m3_checkpoint}")

    # --- 2. 教师模型 (CLIP + MSBDN) ---
    print("\n[2] 加载CLIP雾提示提取器 + HazeCLIP教师...")
    clip_prompt_extractor = CLIPFogPrompt(
        prompt_channels=config.prompt_channels, device=config.device
    )
    teacher = HazeCLIPTeacher(device=config.device)
    print("  教师模型就绪")

    # --- 3. ITL损失 ---
    print("\n[3] 初始化ITL损失...")
    itl_loss_fn = ITLLoss(
        lambda_region=0.5, lambda_boundary=0.3
    ).to(config.device)
    print("  ITL损失就绪")

    # --- 4. 数据集 ---
    print("\n[4] 加载数据集...")
    train_ds = IceAwareDataset(
        config.train_hazy_dir, config.train_clear_dir,
        config.train_ice_dir, config.patch_size, is_train=True
    )
    val_ds = IceAwareDataset(
        config.val_hazy_dir, config.val_clear_dir,
        config.val_ice_dir, config.patch_size, is_train=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    n_ice = sum(1 for _, _, _, has_ice in train_ds.pairs if has_ice)
    print(f"  训练集: {len(train_ds)} 对 (含冰掩码: {n_ice})")
    print(f"  验证集: {len(val_ds)} 对")

    # --- 5. 优化器 ---
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    scaler = torch.amp.GradScaler('cuda')

    # --- 6. 训练循环 ---
    best_psnr = 0.0
    log_file = open(os.path.join(config.log_dir, "train_log.txt"), "w",
                    encoding="utf-8")

    print(f"\n[5] 开始训练 ({config.epochs} epochs)")
    print(f"  lr={config.lr}, batch={config.batch_size}, "
          f"λ_kd={config.lambda_kd}, λ_itl={config.lambda_itl}")
    print("-" * 70)

    for epoch in range(config.epochs):
        model.train()
        clip_prompt_extractor.proj.train()
        teacher.eval()

        epoch_loss = 0
        epoch_recon = 0
        epoch_kd = 0
        epoch_itl = 0
        epoch_region = 0
        epoch_boundary = 0
        n_batches = 0
        n_with_prompt = 0
        t0 = time.time()

        for i, (hazy, clear, ice) in enumerate(train_loader):
            hazy = hazy.to(config.device, non_blocking=True)
            clear = clear.to(config.device, non_blocking=True)
            ice = ice.to(config.device, non_blocking=True)
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

                # 重建损失
                l1 = F.l1_loss(pred, clear)
                ssim_val = 1.0 - ssim_loss(pred, clear)
                loss_recon = config.lambda_l1 * l1 + config.lambda_ssim * ssim_val

                # KD蒸馏损失
                with torch.no_grad():
                    y_teacher = teacher(hazy)
                loss_kd = config.lambda_kd * F.l1_loss(pred, y_teacher)

                # ITL覆冰损失
                r_loss, b_loss, loss_itl_total = itl_loss_fn(
                    pred.float(), clear.float(), ice.float()
                )
                loss = loss_recon + loss_kd + config.lambda_itl * loss_itl_total

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ssim_loss.reset()

            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_kd += loss_kd.item()
            epoch_itl += loss_itl_total.item()
            epoch_region += r_loss
            epoch_boundary += b_loss
            n_batches += 1

            if (i + 1) % 40 == 0:
                tag = "[P]" if use_prompt else "[N]"
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{config.epochs} "
                      f"[{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"(recon={loss_recon.item():.4f}, "
                      f"kd={loss_kd.item():.4f}, "
                      f"itl={loss_itl_total.item():.4f}) "
                      f"{tag} [{int(elapsed)}s]")

        scheduler.step()

        # 验证
        model.eval()
        val_psnr, val_ssim = validate(
            model, val_loader, config.device, use_prompt=False
        )
        val_psnr_p, val_ssim_p = validate(
            model, val_loader, config.device, use_prompt=True,
            clip_extractor=clip_prompt_extractor
        )
        diff = val_psnr_p - val_psnr

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kd = epoch_kd / n_batches
        avg_itl = epoch_itl / n_batches
        avg_region = epoch_region / n_batches
        avg_boundary = epoch_boundary / n_batches
        elapsed = time.time() - t0

        log_line = (f"Epoch {epoch+1}/{config.epochs} "
                    f"loss={avg_loss:.4f} "
                    f"(recon={avg_recon:.4f}, kd={avg_kd:.4f}, "
                    f"itl={avg_itl:.4f} [r={avg_region:.4f},b={avg_boundary:.4f}]) "
                    f"PSNR={val_psnr:.2f}(nop) {val_psnr_p:.2f}(p) "
                    f"[{diff:+.2f}] "
                    f"SSIM={val_ssim:.4f} "
                    f"[p={n_with_prompt}/{n_batches*config.batch_size}] "
                    f"[{int(elapsed)}s]")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        # 保存最佳
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(config.save_dir, "m4_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "best_psnr": best_psnr,
                "config": {k: v for k, v in config.__dict__.items()
                           if not k.startswith("_")},
            }, save_path)
            print(f"  -> 保存最佳 PSNR={val_psnr:.2f} (nop), "
                  f"{val_psnr_p:.2f} (p)")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config.save_dir, f"m4_epoch{epoch+1}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                       save_path)

    log_file.close()

    print("\n" + "=" * 70)
    print(f"Phase 4 训练完成! 最佳PSNR: {best_psnr:.2f}")
    print(f"  M1基线: 34.79, M3: 35.22, M4: {best_psnr:.2f}")
    print(f"  检查点: {os.path.join(config.save_dir, 'm4_best.pth')}")
    print("=" * 70)


if __name__ == "__main__":
    train()
