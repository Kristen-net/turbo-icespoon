"""
WDMamba 独立推理脚本
基于小波变换 + Mamba 选择性扫描的图像去雾
适用于 RTX 5060 8GB 显存
"""
import os
import sys
import glob
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 添加 WDMamba 目录到 sys.path
WDMAMBA_DIR = r"D:\dehaze_fusion\WDMamba"
sys.path.insert(0, WDMAMBA_DIR)

# 测试数据目录和输出目录
TEST_DATA_DIR = r"D:\dehaze_fusion\HazeCLIP\images"
OUTPUT_DIR = os.path.join(WDMamba_DIR if False else r"D:\dehaze_fusion\WDMamba", "output", "wdmamba_results")

# 权重文件 - 使用 Haze4K 权重 (PSNR 35.88, 通用场景最佳)
WEIGHT_PATH = os.path.join(WDMamba_DIR if False else WDMAMBA_DIR, "weights", "WDMamba_ckpts", "haze4k_35.88.pth")

# 备选权重 (RESIDE-6K, PSNR 32.15)
WEIGHT_PATH_RESIDE = os.path.join(WDMAMBA_DIR, "weights", "WDMamba_ckpts", "reside6k_32.15.pth")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor[[2, 1, 0], :, :]  # RGB -> BGR
    tensor = tensor.clamp(0, 1)
    img = (tensor.numpy() * 255).astype(np.uint8)
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

    model.load_state_dict(state_dict, strict=True)
    print("权重加载成功 (strict=True)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e6:.3f} M")
    return model


def main():
    print("=" * 60)
    print("WDMamba 独立推理脚本")
    print("基于小波变换 + Mamba 选择性扫描去雾")
    print("=" * 60)

    # 1. 创建并加载模型
    print("\n[1/4] 创建模型...")
    model = create_model()
    model = load_weights(model, WEIGHT_PATH)
    model.eval()

    # 2. 获取测试图像列表
    print("\n[2/4] 查找测试图像...")
    # 使用 HazeCLIP 的测试图像 + 之前复制到 DehazeSB 的测试图像
    test_dirs = [
        r"D:\dehaze_fusion\HazeCLIP\images",
        r"D:\dehaze_fusion\DehazeSB\test_data"
    ]

    image_paths = set()
    for test_dir in test_dirs:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for p in glob.glob(os.path.join(test_dir, ext)):
                # 只取顶层文件，不递归子目录
                if os.path.dirname(p) == test_dir:
                    image_paths.add(p)

    image_paths = sorted(image_paths)

    # 过滤掉太大的图像（避免 OOM）
    filtered_paths = []
    for p in image_paths:
        size_kb = os.path.getsize(p) / 1024
        if size_kb < 5000:  # 跳过 >5MB 的图像
            filtered_paths.append(p)

    # 只处理前3张测试图像 + 冰检测图像
    final_paths = []
    for p in filtered_paths:
        name = os.path.basename(p)
        if name in ('1.png', '2.png', 'ice1189.jpg') or 'ice' in name.lower():
            final_paths.append(p)
    # 如果没有匹配到，使用前3张
    if not final_paths:
        final_paths = filtered_paths[:3]

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

        print(f"  输入尺寸: {img.shape[1]}x{img.shape[0]}")

        # 转换为tensor
        img_tensor = img2tensor(img).to(device)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        b, c, h, w = img_tensor.size()
        original_h, original_w = h, w

        # 确保尺寸是4的倍数
        img_tensor = check_image_size(img_tensor, window_size=4)
        print(f"  padding后尺寸: {img_tensor.shape}")

        # 使用 bf16 减少显存
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            tm = time.time()
            output = model.restoration_network(img_tensor)
            elapsed = time.time() - tm

        total_time += elapsed
        print(f"  推理时间: {elapsed:.3f}s")

        # 裁剪回原始尺寸
        output = output[:, :, :h, :w]
        print(f"  输出尺寸: {output.shape}")

        # 转换并保存
        output_img = tensor2img(output.float())
        save_name = f"{os.path.splitext(img_name)[0]}_wdmamba_dehazed.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, output_img)
        print(f"  已保存: {save_name}")

        # 检查显存
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  显存使用: {mem_allocated:.2f} GB / {mem_reserved:.2f} GB")

    # 4. 总结
    print("\n" + "=" * 60)
    print(f"[4/4] 推理完成!")
    print(f"  处理图像数: {len(image_paths)}")
    print(f"  总推理时间: {total_time:.2f}s")
    print(f"  平均每张: {total_time / max(len(image_paths), 1):.2f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
