"""
DiffDehaze-GAN DCP 推理脚本
使用暗通道先验 (Dark Channel Prior) 进行物理去雾
无需预训练权重，基于大气散射模型

模型: J = (I - A) / T + A
  J: 清晰图像
  I: 雾图
  A: 大气光
  T: 透射率
"""
import os
import sys
import glob
import time
import cv2
import torch
import numpy as np

# 添加 DiffDehaze-GAN 目录到 sys.path
DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN"
sys.path.insert(0, DIFFDEHAZE_DIR)

# 输出目录
OUTPUT_DIR = os.path.join(DIFFDEHAZE_DIR, "output", "diffdehaze_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像缩放尺寸
RESIZE_SIZE = 256

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def create_model():
    """创建 DCP 去雾生成器"""
    from Model.DCP.DCP_G import DCPDehazeGenerator

    model = DCPDehazeGenerator(
        win_size=5,   # 暗通道窗口大小
        r=15,          # 引导滤波器半径
        eps=1e-3       # 引导滤波器正则化
    ).to(device)
    return model


def img2tensor(img):
    """将 BGR numpy 图像转换为 tensor (1, 3, H, W), 范围 [-1, 1]"""
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    img = img * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    return img


def tensor2img(tensor):
    """将 tensor 从 [0, 1] 转换为 BGR numpy 图像 [0, 255]"""
    tensor = tensor.squeeze(0).cpu()  # (3, H, W)
    tensor = tensor[[2, 1, 0], :, :]  # RGB -> BGR
    tensor = tensor.clamp(0, 1)
    img = (tensor.contiguous().numpy() * 255).astype(np.uint8)  # (3, H, W)
    img = img.transpose(1, 2, 0)  # (H, W, C) for cv2
    return img


def main():
    print("=" * 60)
    print("DiffDehaze-GAN DCP 推理脚本")
    print("基于暗通道先验 (Dark Channel Prior) 物理去雾")
    print("=" * 60)

    # 1. 创建模型 (无参数)
    print("\n[1/4] 创建 DCP 模型...")
    model = create_model()
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params} (DCP 无可训练参数)")

    # 2. 获取测试图像
    print("\n[2/4] 查找测试图像...")
    test_dir = r"D:\dehaze_fusion\HazeCLIP\images"
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for p in glob.glob(os.path.join(test_dir, ext)):
            if os.path.dirname(p) == test_dir:
                image_paths.append(p)
    image_paths = sorted(image_paths)

    if not image_paths:
        print("错误: 未找到测试图像")
        return

    print(f"找到 {len(image_paths)} 张测试图像:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)} ({os.path.getsize(p) / 1024:.1f} KB)")

    # 3. 推理
    print(f"\n[3/4] 开始推理...")
    total_time = 0

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"\n--- 处理 [{idx + 1}/{len(image_paths)}]: {img_name} ---")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("  跳过: 无法读取图像")
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = img.shape[:2]
        print(f"  原始尺寸: {orig_w}x{orig_h}")

        if RESIZE_SIZE > 0:
            img = cv2.resize(img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_AREA)
            print(f"  缩放至: {img.shape[1]}x{img.shape[0]}")

        # 转换为 tensor [-1, 1]
        img_tensor = img2tensor(img).to(device).unsqueeze(0)

        print(f"  输入 tensor: {img_tensor.shape}, range [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        try:
            with torch.no_grad():
                tm = time.time()
                J_DCP, T_DCP, A = model(img_tensor)
                elapsed = time.time() - tm

            total_time += elapsed
            print(f"  推理时间: {elapsed:.3f}s")
            print(f"  大气光 A: {A.cpu().numpy()}")
            print(f"  透射率 T range: [{T_DCP.min():.4f}, {T_DCP.max():.4f}]")
            print(f"  去雾结果 J range: [{J_DCP.min():.4f}, {J_DCP.max():.4f}]")

            # 保存去雾结果
            dehazed_img = tensor2img(J_DCP.float())
            save_name = f"{os.path.splitext(img_name)[0]}_dcp_dehazed.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, dehazed_img)
            print(f"  已保存: {save_name} ({os.path.getsize(save_path) / 1024:.1f} KB)")

            # 保存透射率图 (可视化)
            T_vis = T_DCP.squeeze().cpu().numpy()
            T_vis = (T_vis * 255).astype(np.uint8)
            T_vis = cv2.applyColorMap(T_vis, cv2.COLORMAP_JET)
            T_name = f"{os.path.splitext(img_name)[0]}_transmission.png"
            T_path = os.path.join(OUTPUT_DIR, T_name)
            cv2.imwrite(T_path, T_vis)
            print(f"  透射率图: {T_name}")

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
    print(f"  平均每张: {total_time / max(len(image_paths), 1):.3f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
