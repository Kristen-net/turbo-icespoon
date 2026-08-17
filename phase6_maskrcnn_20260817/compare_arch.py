import torch
import torchvision

print("=== torchvision Mask R-CNN keys ===")
from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(weights=None, num_classes=2, weights_backbone=None)
tv_keys = list(model.state_dict().keys())
print(f"Total keys: {len(tv_keys)}")
for k in tv_keys[:50]:
    print(f"  {k}: {model.state_dict()[k].shape}")
print("--- last 20 ---")
for k in tv_keys[-20:]:
    print(f"  {k}: {model.state_dict()[k].shape}")

print("\n=== detectron2 checkpoint keys ===")
path = r'c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth'
ckpt = torch.load(path, map_location='cpu', weights_only=False)
d2_keys = list(ckpt['model'].keys())
print(f"Total keys: {len(d2_keys)}")

# Group by prefix
prefixes = {}
for k in d2_keys:
    prefix = '.'.join(k.split('.')[:3])
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(k)

print("\n=== detectron2 key groups ===")
for p in sorted(prefixes.keys()):
    print(f"  {p}: {len(prefixes[p])} params")

# Group torchvision keys by prefix too
tv_prefixes = {}
for k in tv_keys:
    prefix = '.'.join(k.split('.')[:3])
    if prefix not in tv_prefixes:
        tv_prefixes[prefix] = []
    tv_prefixes[prefix].append(k)

print("\n=== torchvision key groups ===")
for p in sorted(tv_prefixes.keys()):
    print(f"  {p}: {len(tv_prefixes[p])} params")
