"""测试不同输入配置下的Mask R-CNN推理"""
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

CKPT_PATH = r"c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth"
DEVICE = "cuda"

def load_model(swap_conv1=False, image_mean=None, image_std=None):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from maskrcnn_inference import build_mapping

    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    d2_sd = ckpt['model']

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=2, weights_backbone=None,
                                  min_size=800, max_size=1333,
                                  box_score_thresh=0.05, box_detections_per_img=100)

    if image_mean:
        model.transform.image_mean = image_mean
    if image_std:
        model.transform.image_std = image_std

    tv_sd = model.state_dict()
    mapping = build_mapping(list(d2_sd.keys()), list(tv_sd.keys()))

    new_sd = {}
    for d2k, tvk in mapping.items():
        if tvk in tv_sd:
            d2_val = d2_sd[d2k]
            tv_val = tv_sd[tvk]
            if d2_val.shape == tv_val.shape:
                new_sd[tvk] = d2_val.clone()
            else:
                if 'bbox_pred' in tvk:
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                elif 'mask_fcn_logits' in tvk:
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                else:
                    print(f"  MISMATCH: {d2k} {d2_val.shape} -> {tvk} {tv_val.shape}")

    for tvk in tv_sd:
        if tvk not in new_sd and 'num_batches_tracked' in tvk:
            new_sd[tvk] = torch.tensor(0, dtype=torch.long)

    # 可选: 交换conv1通道 (BGR→RGB)
    if swap_conv1 and 'backbone.body.conv1.weight' in new_sd:
        w = new_sd['backbone.body.conv1.weight']
        new_sd['backbone.body.conv1.weight'] = w[:, [2, 1, 0], :, :].clone()
        print("  已交换conv1通道 (BGR→RGB)")

    model.load_state_dict(new_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def test_config(model, img_bgr, config_name):
    h, w = img_bgr.shape[:2]
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float().to(DEVICE)

    with torch.no_grad():
        outputs = model([img_tensor])

    pred = outputs[0]
    scores = pred['scores'].cpu().numpy()
    n = len(scores)
    if n > 0:
        score_range = f"[{scores.min():.4f}, {scores.max():.4f}]"
        n_95 = np.sum(scores >= 0.95)
        n_99 = np.sum(scores >= 0.99)
        boxes = pred['boxes'].cpu().numpy()
        valid = np.sum((boxes[:, 2] - boxes[:, 0] > 5) & (boxes[:, 3] - boxes[:, 1] > 5))
        print(f"  {config_name}: {n} dets, scores {score_range}, >=0.95:{n_95}, >=0.99:{n_99}, valid_boxes:{valid}")
    else:
        print(f"  {config_name}: 0 detections")


img_path = r"D:\dehaze_fusion\my_test\input\real_0002.png"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Image: {img_path}, shape: {img_bgr.shape}")
print()

# Config 1: BGR输入, detectron2 BGR mean (原始配置)
print("=== Config 1: BGR + D2 mean (BGR) ===")
model1 = load_model(swap_conv1=False, image_mean=[103.530, 116.280, 123.675], image_std=[1.0, 1.0, 1.0])
test_config(model1, img_bgr, "BGR + BGR_mean")

# Config 2: RGB输入, detectron2 RGB mean (反转)
print("\n=== Config 2: RGB + D2 mean (RGB, reversed) ===")
model2 = load_model(swap_conv1=True, image_mean=[123.675, 116.280, 103.530], image_std=[1.0, 1.0, 1.0])
test_config(model2, img_rgb, "RGB + RGB_mean")

# Config 3: RGB输入, 不交换conv1, ImageNet mean
print("\n=== Config 3: RGB + ImageNet mean (no swap) ===")
model3 = load_model(swap_conv1=False, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
test_config(model3, img_rgb, "RGB + ImageNet_mean")

# Config 4: BGR输入, 交换conv1, ImageNet mean (BGR)
print("\n=== Config 4: BGR + swap + ImageNet mean (BGR) ===")
model4 = load_model(swap_conv1=True, image_mean=[0.406, 0.456, 0.485], image_std=[0.225, 0.224, 0.229])
test_config(model4, img_bgr, "BGR + ImageNet_BGR_mean")

# Config 5: 不归一化
print("\n=== Config 5: No normalization ===")
model5 = load_model(swap_conv1=False, image_mean=[0, 0, 0], image_std=[1.0, 1.0, 1.0])
test_config(model5, img_bgr, "BGR + no_norm")

# Config 6: 检查原始detectron2的pixel_mean值
print("\n=== Check checkpoint for normalization clues ===")
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
trainer = ckpt.get('trainer', {})
if isinstance(trainer, dict):
    for k, v in trainer.items():
        if k != '_trainer':
            print(f"  trainer.{k}: {str(v)[:200]}")
