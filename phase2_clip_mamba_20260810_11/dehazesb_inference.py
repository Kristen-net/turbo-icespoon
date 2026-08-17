"""
DehazeSB 独立推理脚本
直接加载 DSCNet_unetgan 生成器，绕过 vision_aided_loss 等训练专用依赖
实现多步 Schrödinger Bridge 扩散去雾前向传播
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import tqdm

# 添加 DehazeSB 目录到 sys.path
DEHAZESB_DIR = r"D:\dehaze_fusion\DehazeSB"
sys.path.insert(0, DEHAZESB_DIR)

# 测试数据目录和输出目录
TEST_DATA_DIR = os.path.join(DEHAZESB_DIR, "test_data")
OUTPUT_DIR = os.path.join(DEHAZESB_DIR, "output", "dehazesb_results")
WEIGHT_PATH = os.path.join(DEHAZESB_DIR, "pretrained", "pretrained", "net_G.pth")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def create_model():
    """创建 DSCNet_unetgan 生成器"""
    from models.DSCNet import DSCNet_unetgan

    # 参数与 base_options.py 中的默认值一致
    model = DSCNet_unetgan(
        n_channels=3,          # input_nc
        n_classes=3,           # output_nc
        kernel_size=9,          # DSC_kernel
        extend_scope=1.0,      # DSC_extend
        if_offset=True,         # DSC_offset
        device=device,
        number=32,              # DSC_number
        dim=1,
        n_blocks=9,             # DSC_n_blocks
        padding_type='reflect', # DSE_padding_type
        use_dropout=False       # not no_dropout (no_dropout=True)
    )
    return model


def load_weights(model, weight_path):
    """加载预训练权重"""
    print(f"加载权重: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)

    # 检查并处理可能的键名前缀
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # 尝试直接加载
    try:
        model.load_state_dict(state_dict, strict=True)
        print("权重加载成功 (strict=True)")
    except RuntimeError as e:
        print(f"strict=True 失败, 尝试 strict=False: {e}")
        # 尝试去掉 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功 (strict=False)")

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e6:.3f} M")
    return model


def get_times(T=5):
    """计算 Schrödinger Bridge 时间步"""
    tau = 0.01  # opt.tau
    incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    times = torch.tensor(times).float().to(device)
    return times, tau


def forward_diffusion_dehaze(model, real_A, num_timesteps=5, ngf=64):
    """
    多步 Schrödinger Bridge 扩散去雾前向传播
    与 dehazesbtest_model.py 中的 test phase forward 一致

    参数:
        model: DSCNet_unetgan 生成器
        real_A: 输入雾图 tensor (B, 3, H, W), 范围 [-1, 1]
        num_timesteps: 扩散步数 (默认5)
        ngf: 生成器滤波器数 (默认64)

    返回:
        outputs: 各步骤的输出列表 [fake_1, fake_2, ..., fake_T]
    """
    times, tau = get_times(num_timesteps)
    outputs = []
    model.eval()

    with torch.no_grad():
        for t in tqdm.tqdm(range(num_timesteps), desc="扩散去雾"):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                # 添加噪声的扩散步骤
                Xt = (1 - inter) * Xt + inter * Xt_1.detach() + \
                     (scale * tau).sqrt() * torch.randn_like(Xt).to(device)
            else:
                Xt = real_A

            time_idx = (t * torch.ones(size=[real_A.shape[0]]).to(device)).long()
            z = torch.randn(size=[real_A.shape[0], 4 * ngf]).to(device)

            # 生成器前向传播
            Xt_1, time_embed = model(Xt, time_idx, z)
            outputs.append(Xt_1)

    return outputs


def preprocess_image(image_path, max_size=1024):
    """
    预处理图像
    - 大图缩放到 max_size x max_size
    - 小图调整为 8 的倍数
    - 归一化到 [-1, 1]
    """
    image = Image.open(image_path).convert("RGB")
    if image.height >= 1000 or image.width >= 1000:
        image = image.resize((max_size, max_size), Image.LANCZOS)
    else:
        new_h = (image.height // 8) * 8
        new_w = (image.width // 8) * 8
        image = image.resize((new_w, new_h), Image.LANCZOS)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image


def tensor_to_pil(tensor):
    """将 tensor 从 [-1, 1] 转换为 PIL 图像"""
    tensor = tensor[0].cpu() * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return T.ToPILImage()(tensor)


def main():
    print("=" * 60)
    print("DehazeSB 独立推理脚本")
    print("基于 Schrödinger Bridge 扩散去雾")
    print("=" * 60)

    # 1. 创建并加载模型
    print("\n[1/4] 创建模型...")
    model = create_model()
    model = load_weights(model, WEIGHT_PATH)
    model = model.to(device)

    # 2. 获取测试图像列表
    print("\n[2/4] 查找测试图像...")
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(TEST_DATA_DIR, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"错误: 在 {TEST_DATA_DIR} 中未找到测试图像")
        return

    print(f"找到 {len(image_paths)} 张测试图像:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    # 3. 逐张推理
    print(f"\n[3/4] 开始推理 (num_timesteps=5)...")
    total_time = 0

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"\n--- 处理 [{idx + 1}/{len(image_paths)}]: {img_name} ---")

        # 预处理
        input_tensor, original_image = preprocess_image(img_path)
        print(f"  输入尺寸: {original_image.size} -> tensor: {input_tensor.shape}")

        # 使用 bf16 减少显存占用 (RTX 5060 8GB)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = forward_diffusion_dehaze(
                model, input_tensor, num_timesteps=5, ngf=64
            )

        # 保存各步骤结果
        for step, output in enumerate(outputs):
            step_name = f"fake_{step + 1}"
            pil_img = tensor_to_pil(output.float())
            save_name = f"{os.path.splitext(img_name)[0]}_{step_name}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            pil_img.save(save_path)
            print(f"  {step_name} 已保存: {save_name} ({pil_img.size})")

        # 保存最终结果 (最后一步)
        final_output = outputs[-1]
        final_pil = tensor_to_pil(final_output.float())
        final_name = f"{os.path.splitext(img_name)[0]}_dehazed.png"
        final_path = os.path.join(OUTPUT_DIR, final_name)
        final_pil.save(final_path)
        print(f"  最终去雾结果: {final_name}")

        # 检查显存使用
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  显存使用: {mem_allocated:.2f} GB / {mem_reserved:.2f} GB")

    # 4. 总结
    print("\n" + "=" * 60)
    print(f"[4/4] 推理完成!")
    print(f"  处理图像数: {len(image_paths)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  每张图像生成 5 个中间步骤 + 1 个最终结果")
    print("=" * 60)


if __name__ == "__main__":
    main()
