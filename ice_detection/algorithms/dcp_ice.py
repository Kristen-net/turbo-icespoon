"""
DCP 去雾 - 单图推理 (ice1189.jpg)
"""
import os
import sys
import cv2
import torch
import numpy as np

DIFFDEHAZE_DIR = r"D:\dehaze_fusion\DiffDehaze-GAN"
sys.path.insert(0, DIFFDEHAZE_DIR)

from Model.DCP.DCP_G import DCPDehazeGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DCPDehazeGenerator().to(device)
model.eval()

# Load ice1189
img_path = r"D:\dehaze_fusion\HazeCLIP\images\ice1189.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# Preprocess: BGR->RGB, [0,255]->[-1,1], HWC->CHW->NCHW
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    J_DCP, T_DCP, A = model(img_tensor)

# J_DCP is in [0,1] range (forward uses (x+1)/2 internally)
J_np = J_DCP.squeeze(0).permute(1, 2, 0).cpu().numpy()
J_np = np.clip(J_np, 0, 1)
J_bgr = cv2.cvtColor((J_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Transmission map
T_np = T_DCP.squeeze(0).squeeze(0).cpu().numpy()
T_vis = (T_np * 255).astype(np.uint8)
T_color = cv2.applyColorMap(T_vis, cv2.COLORMAP_JET)

# Save
out_dir = os.path.join(DIFFDEHAZE_DIR, "output", "diffdehaze_results")
os.makedirs(out_dir, exist_ok=True)

cv2.imwrite(os.path.join(out_dir, "ice1189_dcp_dehazed.png"), J_bgr)
cv2.imwrite(os.path.join(out_dir, "ice1189_transmission.png"), T_color)

print(f"DCP dehazed saved: {os.path.join(out_dir, 'ice1189_dcp_dehazed.png')}")
print(f"Transmission saved: {os.path.join(out_dir, 'ice1189_transmission.png')}")
print(f"Atmospheric light A: {A.cpu().numpy()}")
