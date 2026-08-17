"""
WDMamba 独立推理脚本 v2
- 支持图像缩放（默认 256x256 用于快速测试）
- 详细的进度和时间输出
- 适用于 RTX 5060 8GB 显存
"""
import os
import sys
import glob
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# 添加 WDMamba 目录到 sys.path
WDMAMBA_DIR = r"D:\dehaze_fusion\WDMamba"
sys.path.insert(0, WDMAMBA_DIR)

# 权重文件 - 使用 Haze4K 权重
WEIGHT_PATH = os.path.join(WDMAMBA_DIR, "weights", "WDMamba_ckpts", "haze4k_35.88.pth")

# 输出目录
OUTPUT_DIR = os.path.join(WDMAMBA_DIR, "output", "wdmamba_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像缩放尺寸（0 = 不缩放）
RESIZE_SIZE = 256

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def check_image_size(x, window_size=4):
    """确保图像尺寸是 window_size 的倍数"""
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def img2tensor(img):
    """将 BGR numpy 图像转换为 tensor (C, H, W), 范围 [0, 1]"""
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def tensor2img(tensor):
    """将 tensor 转换为 BGR numpy 图像, 范围 [0, 255]"""
    tensor = tensor.squeeze(0).cpu()  # (3, H, W)
    tensor = tensor[[2, 1, 0], :, :]  # RGB -> BGR
    tensor = tensor.clamp(0, 1)
    img = (tensor.contiguous().numpy() * 255).astype(np.uint8)  # (3, H, W)
    img = img.transpose(1, 2, 0)  # (H, W, C) for cv2
    return img


def create_model():
    """创建 WaveMamba 模型"""
    from basicsr.archs.wavemamba_arch import WaveMamba

    model = WaveMamba(
        in_chn=3,
        wf=16,
        n_l_blocks=[1, 2, 2, 4],
        ffn_scale=2.0
    ).to(device)

    return model


def load_weights(model, weight_path):
    """加载预训练权重"""
    print(f"加载权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)

    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # 尝试 strict 加载
    try:
        model.load_state_dict(state_dict, strict=True)
        print("权重加载成功 (strict=True)")
    except RuntimeError as e:
        print(f"strict=True 失败, 尝试 strict=False...")
        # 去掉可能的 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功 (strict=False)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e6:.3f} M")
    return model


def main():
    print("=" * 60)
    print("WDMamba 独立推理脚本 v2")
    print("基于小波变换 + Mamba 选择性扫描去雾")
    if RESIZE_SIZE > 0:
        print(f"测试模式: 图像缩放至 {RESIZE_SIZE}x{RESIZE_SIZE}")
    print("=" * 60)

    # 1. 创建并加载模型
    print("\n[1/4] 创建模型...")
    t0 = time.time()
    model = create_model()
    model = load_weights(model, WEIGHT_PATH)
    model.eval()
    print(f"模型准备完成, 耗时 {time.time() - t0:.1f}s")

    # 2. 获取测试图像列表
    print("\n[2/4] 查找测试图像...")
    # 只从 HazeCLIP 目录取图，避免重复
    test_dir = r"D:\dehaze_fusion\HazeCLIP\images"

    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for p in glob.glob(os.path.join(test_dir, ext)):
            if os.path.dirname(p) == test_dir:
                image_paths.append(p)

    image_paths = sorted(image_paths)

    # 只处理前2张测试图像
    final_paths = image_paths[:2]

    image_paths = final_paths

    if not image_paths:
        print("错误: 未找到测试图像")
        return

    print(f"找到 {len(image_paths)} 张测试图像:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)} ({os.path.getsize(p) / 1024:.1f} KB)")

    # 3. 逐张推理
    print(f"\n[3/4] 开始推理...")
    total_time = 0

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"\n--- 处理 [{idx + 1}/{len(image_paths)}]: {img_name} ---")

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  跳过: 无法读取图像")
            continue

        # 确保是3通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = img.shape[:2]
        print(f"  原始尺寸: {orig_w}x{orig_h}")

        # 缩放图像（如果设置了 RESIZE_SIZE）
        if RESIZE_SIZE > 0:
            img = cv2.resize(img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_AREA)
            print(f"  缩放至: {img.shape[1]}x{img.shape[0]}")

        # 转换为tensor
        img_tensor = img2tensor(img).to(device)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        b, c, h, w = img_tensor.size()

        # 确保尺寸是4的倍数
        img_tensor = check_image_size(img_tensor, window_size=4)
        print(f"  padding后尺寸: {img_tensor.shape}")

        # 清空缓存
        torch.cuda.empty_cache()

        # 推理
        try:
            print(f"  开始 Mamba 前向传播...")
            with torch.no_grad():
                tm = time.time()
                output = model.restoration_network(img_tensor)
                elapsed = time.time() - tm

            total_time += elapsed
            print(f"  推理时间: {elapsed:.2f}s")

            # 检查输出是否有效
            print(f"  输出 dtype: {output.dtype}")
            print(f"  输出 min/max: {output.min().item():.6f} / {output.max().item():.6f}")
            print(f"  输出 mean: {output.mean().item():.6f}")
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                print(f"  警告: 输出包含 {nan_count} 个 NaN!")
                output = torch.nan_to_num(output, nan=0.0)
            if torch.isinf(output).any():
                inf_count = torch.isinf(output).sum().item()
                print(f"  警告: 输出包含 {inf_count} 个 Inf!")
                output = torch.nan_to_num(output, posinf=1.0, neginf=0.0)

            # 裁剪回原始尺寸
            output = output[:, :, :h, :w]
            print(f"  输出尺寸: {output.shape}")

            # 转换并保存
            output_img = tensor2img(output.float())
            print(f"  numpy 图像 shape: {output_img.shape}, dtype: {output_img.dtype}")
            print(f"  numpy 图像 min/max: {output_img.min()} / {output_img.max()}")
            save_name = f"{os.path.splitext(img_name)[0]}_wdmamba_dehazed.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            write_ret = cv2.imwrite(save_path, output_img)
            print(f"  cv2.imwrite 返回: {write_ret}")
            print(f"  文件存在: {os.path.exists(save_path)}")
            if write_ret and os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / 1024
                print(f"  已保存: {save_name} ({file_size:.1f} KB)")
            else:
                print(f"  保存失败! 尝试备用路径...")
                backup_path = os.path.join(OUTPUT_DIR, f"backup_{save_name}")
                cv2.imwrite(backup_path, output_img)
                print(f"  备用保存: {os.path.exists(backup_path)}")

            # 检查显存
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  显存使用: {mem_allocated:.2f} GB / {mem_reserved:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  错误: 显存不足 (OOM)")
                print(f"  建议: 减小 RESIZE_SIZE 或使用分块推理")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"  错误: {e}")
                continue
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. 总结
    print("\n" + "=" * 60)
    print(f"[4/4] 推理完成!")
    print(f"  处理图像数: {len(image_paths)}")
    print(f"  总推理时间: {total_time:.2f}s")
    if len(image_paths) > 0:
        print(f"  平均每张: {total_time / len(image_paths):.2f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
