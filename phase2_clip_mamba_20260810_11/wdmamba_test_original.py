"""
WDMamba 原始推理方式测试
直接使用 basicsr 的 img2tensor/tensor2img，不缩放图像
"""
import os
import sys
import cv2
import torch
import torch.nn.functional as F

WDMAMBA_DIR = r"D:\dehaze_fusion\WDMamba"
sys.path.insert(0, WDMAMBA_DIR)

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.wavemamba_arch import WaveMamba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_PATH = os.path.join(WDMAMBA_DIR, "weights", "WDMamba_ckpts", "haze4k_35.88.pth")
OUTPUT_DIR = os.path.join(WDMAMBA_DIR, "output", "wdmamba_original_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_image_size(x, window_size=4):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

model = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1,2,2,4], ffn_scale=2.0).to(device)
checkpoint = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint['params'], strict=True)
model.eval()
print("Model loaded, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# Test with 1.png at original size
test_path = r"D:\dehaze_fusion\HazeCLIP\images\1.png"
img = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
print(f"Image shape: {img.shape}")

# Original preprocessing
img_tensor = img2tensor(img).to(device) / 255.
img_tensor = img_tensor.unsqueeze(0)
b, c, h, w = img_tensor.size()
print(f"Tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

img_tensor = check_image_size(img_tensor)

with torch.no_grad():
    output = model.restoration_network(img_tensor)

output = output[:, :, :h, :w]
print(f"Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")

output_img = tensor2img(output)
print(f"Output img shape: {output_img.shape}, dtype: {output_img.dtype}")

save_path = os.path.join(OUTPUT_DIR, "1_original_preproc.png")
imwrite(output_img, save_path)
print(f"Saved: {save_path}")
