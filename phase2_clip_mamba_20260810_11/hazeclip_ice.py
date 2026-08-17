"""
HazeCLIP 单图推理脚本 - 用于 ice1189.jpg
"""
import os
import sys
import cv2
import torch
import numpy as np

HazeCLIP_DIR = r"D:\dehaze_fusion\HazeCLIP"
sys.path.insert(0, HazeCLIP_DIR)
sys.path.insert(0, os.path.join(HazeCLIP_DIR, "CLIP"))

from model import HazeCLIPModel
from CLIP import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = HazeCLIPModel()
checkpoint = torch.load(os.path.join(HazeCLIP_DIR, "weights", "model.pth"), map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

# Load CLIP
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()

# Text prompt
text = clip.tokenize(["hazy image", "clear image"]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Process ice1189.jpg
img_path = os.path.join(HazeCLIP_DIR, "images", "ice1189.jpg")
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize for CLIP (224x224)
h, w = img_rgb.shape[:2]
img_clip = cv2.resize(img_rgb, (224, 224))

# Normalize
def normalize(img, mean, std):
    return (img - mean) / std

img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
img_clip_tensor = torch.from_numpy(normalize(img_clip.astype(np.float32)/255.0, 
    [0.48145466, 0.4578275, 0.40821073],
    [0.26862954, 0.26130258, 0.27577711])).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    # CLIP visual features
    clip_feat = clip_model.encode_image(img_clip_tensor)
    clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
    
    # Similarity
    sim = (clip_feat @ text_features.T).squeeze(0)
    weight = torch.softmax(sim, dim=0)[1]  # weight for "clear"
    print(f"Haze confidence: {sim[0].item():.4f}, Clear confidence: {sim[1].item():.4f}")
    print(f"Dehaze weight: {weight.item():.4f}")
    
    # Dehaze
    output = model(img_tensor, clip_feat)
    output = torch.clamp(output, 0, 1)

# Save
output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

out_dir = os.path.join(HazeCLIP_DIR, "outputs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ice1189.jpg")
cv2.imwrite(out_path, output_bgr)
print(f"Saved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
