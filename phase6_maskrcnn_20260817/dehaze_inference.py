"""
IceWave-DehazeFormer 图片去雾推理脚本 (含YOLO自动训练)

使用方法:
  1. 单张图片:
     python dehaze_inference.py -i D:\photos\foggy.jpg -o D:\photos\clear.jpg

  2. 批量处理文件夹:
     python dehaze_inference.py -i D:\photos\hazy\ -o D:\photos\dehazed\

  3. 同时生成覆冰掩码和对比图 (推荐):
     python dehaze_inference.py -i D:\photos\input\ -o D:\photos\output\ --ice-mask --compare

  4. 指定模型 (默认M4, 性能最佳):
     python dehaze_inference.py -i D:\photos\input\ -o D:\photos\output\ -m m3 --compare

放入input目录的新图片会自动触发YOLO重训 (增量数据 + 迁移学习)

可选模型: m1(基线), m2(HA-WFE v1), m2p(HA-WFE v2), m3(CLIP蒸馏), m4(ITL覆冰感知, 默认)
"""

import sys
import os
import argparse
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision.ops import nms as tv_nms

sys.path.insert(0, r"D:\dehaze_fusion\DehazeFormer")
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

from models.dehazeformer import dehazeformer_s
from ha_wfe import integrate_hawfe
from ha_wfe_v2 import integrate_hawfe_v2
from clip_fog_prompt import integrate_hawfe_v2_with_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = {
    "m1":  r"D:\dehaze_fusion\icewave_output\checkpoints\m1_best.pth",
    "m2":  r"D:\dehaze_fusion\icewave_output\m2_hawfe\checkpoints\m2_best.pth",
    "m2p": r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\checkpoints\m2p_best.pth",
    "m3":  r"D:\dehaze_fusion\icewave_output\m3_clip_distill\checkpoints\m3_best.pth",
    "m4":  r"D:\dehaze_fusion\icewave_output\m4_itl\checkpoints\m4_best.pth",
}

YOLO_PATH = r"D:\dehaze_fusion\yolo_train_output\power_line_yolo\weights\best.pt"
YOLO_CLASSES = ['insulator', 'power_line', 'ice', 'tower']
YOLO_COLORS = {
    'insulator': (0, 165, 255),
    'power_line': (255, 0, 0),
    'ice': (0, 0, 255),
    'tower': (0, 255, 0),
}

YOLO_DATASET_DIR = r"D:\dehaze_fusion\yolo_dataset"
YOLO_TRAIN_IMG = os.path.join(YOLO_DATASET_DIR, "images", "train")
YOLO_TRAIN_LBL = os.path.join(YOLO_DATASET_DIR, "labels", "train")
YOLO_VAL_IMG = os.path.join(YOLO_DATASET_DIR, "images", "val")
YOLO_VAL_LBL = os.path.join(YOLO_DATASET_DIR, "labels", "val")
YOLO_MANIFEST = os.path.join(YOLO_DATASET_DIR, "manifest.txt")
YOLO_DATA_YAML = os.path.join(YOLO_DATASET_DIR, "data.yaml")
YOLO_OUTPUT_DIR = r"D:\dehaze_fusion\yolo_train_output\power_line_yolo"

MODEL_DESC = {
    "m1":  "DehazeFormer-S 基线",
    "m2":  "+ HA-WFE v1 (零初始化, Tanh, 共享alpha)",
    "m2p": "+ HA-WFE v2 (正值初始化, Sigmoid, 独立alpha)",
    "m3":  "+ CLIP雾提示蒸馏 (推理时无需CLIP)",
    "m4":  "+ ITL覆冰感知损失 (区域+边界约束, 推荐使用)",
}


def load_model(name):
    model = dehazeformer_s().to(DEVICE)
    if name == "m1":
        ckpt = torch.load(CKPT[name], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    elif name == "m2":
        model = integrate_hawfe(model, channels=96)
        ckpt = torch.load(CKPT[name], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    elif name == "m2p":
        model = integrate_hawfe_v2(model, channels=96)
        ckpt = torch.load(CKPT[name], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    elif name in ("m3", "m4"):
        model = integrate_hawfe_v2_with_prompt(model, channels=96, prompt_channels=32)
        ckpt = torch.load(CKPT[name], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
        model.clip_prompt = None
    else:
        raise ValueError(f"未知模型: {name}, 可选: {list(CKPT.keys())}")
    model.eval()
    return model


def dehaze_image(model, img_bgr, name):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0

    if name in ("m3", "m4"):
        model.clip_prompt = None

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        with torch.amp.autocast(DEVICE, dtype=torch.float16):
            pred = model(img_t).float().clamp(0, 1)

    pred = pred[:, :, :h, :w]
    pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)


def _file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    if os.path.exists(YOLO_MANIFEST):
        with open(YOLO_MANIFEST, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_manifest(manifest):
    with open(YOLO_MANIFEST, "w") as f:
        for item in sorted(manifest):
            f.write(item + "\n")


def _auto_label_single(img_path, out_img_path, out_label_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    boxes = []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    mask1 = cv2.inRange(hsv, np.array([5, 30, 40]), np.array([25, 200, 200]))
    mask2 = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 180]))
    ins_mask = cv2.morphologyEx(mask1 | mask2, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    ins_mask = cv2.morphologyEx(ins_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    for cnt in cv2.findContours(ins_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if 500 < area < h * w * 0.3:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw / max(bh, 1) < 2.0 and bh > 20:
                boxes.append((0, x, y, bw, bh))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                           minLineLength=min(h, w) // 4, maxLineGap=30)
    if lines is not None:
        for line in lines:
            pts = line.reshape(-1) if line.ndim > 1 else line
            if len(pts) >= 4:
                x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                if abs(y2 - y1) < 30:
                    boxes.append((1, min(x1, x2), min(y1, y2),
                                  abs(x2 - x1), abs(y2 - y1)))

    low_sat = s < 60
    high_val = v > 160
    white_mask = (low_sat & high_val).astype(np.uint8) * 255
    # 中等梯度纹理, 排除极高梯度 (水印锐边)
    grad_x2 = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y2 = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    grad_mag2 = np.sqrt(grad_x2 ** 2 + grad_y2 ** 2)
    grad_norm2 = (grad_mag2 / (grad_mag2.max() + 1e-6) * 255).astype(np.uint8)
    grad_blur2 = cv2.GaussianBlur(grad_norm2, (5, 5), 0)
    _, texture_mask2 = cv2.threshold(grad_blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, high_grad2 = cv2.threshold(grad_blur2, 180, 255, cv2.THRESH_BINARY)
    texture_mask2 = cv2.subtract(texture_mask2, high_grad2)
    ice_mask = cv2.bitwise_and(white_mask, texture_mask2)
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    for cnt in cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if bw > 20 and bh > 20 and 0.125 < aspect < 8.0:
                boxes.append((2, x, y, bw, bh))

    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).astype(np.uint8)
    _, binary = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    for cnt in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw / max(bh, 1) < 0.8 and bh > h * 0.15:
                boxes.append((3, x, y, bw, bh))

    valid = []
    for cls_id, x, y, bw, bh in boxes:
        x, bw = max(0, x), min(bw, w - max(0, x))
        y, bh = max(0, y), min(bh, h - max(0, y))
        if bw > 10 and bh > 10:
            valid.append((cls_id, x, y, bw, bh))

    cv2.imwrite(out_img_path, img)
    with open(out_label_path, "w") as f:
        for cls_id, x, y, bw, bh in valid:
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            wn, hn = bw / w, bh / h
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    return len(valid)


def check_and_retrain_yolo(input_dir):
    if not os.path.isdir(input_dir):
        return

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])
    if not input_files:
        return

    manifest = _load_manifest()
    new_images = []
    for fname in input_files:
        fpath = os.path.join(input_dir, fname)
        fhash = _file_hash(fpath)
        key = f"{fname}:{fhash}"
        if key not in manifest:
            new_images.append((fname, fpath, key))

    if not new_images:
        print("[YOLO] 无新图片, 跳过训练")
        return

    print(f"[YOLO] 检测到 {len(new_images)} 张新图片, 开始自动标注...")
    for d in [YOLO_TRAIN_IMG, YOLO_TRAIN_LBL, YOLO_VAL_IMG, YOLO_VAL_LBL]:
        os.makedirs(d, exist_ok=True)

    n_train = 0
    for i, (fname, fpath, key) in enumerate(new_images):
        base = os.path.splitext(fname)[0]
        safe_name = base.replace('.rf.', '_').replace('.', '_')
        is_val = (i % 5 == 0)
        img_dir = YOLO_VAL_IMG if is_val else YOLO_TRAIN_IMG
        lbl_dir = YOLO_VAL_LBL if is_val else YOLO_TRAIN_LBL
        out_img = os.path.join(img_dir, f"{safe_name}.jpg")
        out_lbl = os.path.join(lbl_dir, f"{safe_name}.txt")
        n = _auto_label_single(fpath, out_img, out_lbl)
        n_train += n
        manifest.add(key)

    print(f"[YOLO] 标注完成: {len(new_images)} 张图片, {n_train} 个框")

    yaml_content = f"""path: {YOLO_DATASET_DIR.replace(os.sep, '/')}
train: images/train
val: images/val

nc: 4
names: ['insulator', 'power_line', 'ice', 'tower']
"""
    with open(YOLO_DATA_YAML, "w") as f:
        f.write(yaml_content)

    for cache in ["train.cache", "val.cache"]:
        for d in [YOLO_TRAIN_LBL, YOLO_VAL_LBL]:
            cp = os.path.join(d, cache)
            if os.path.exists(cp):
                os.remove(cp)

    print(f"[YOLO] 开始重训 (迁移学习, 20 epochs)...")
    from ultralytics import YOLO
    yolo = YOLO(YOLO_PATH if os.path.exists(YOLO_PATH)
                else r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d\yolov8n.pt")
    yolo.train(
        data=YOLO_DATA_YAML,
        epochs=20,
        imgsz=640,
        batch=8,
        device=0 if DEVICE == "cuda" else "cpu",
        project=r"D:\dehaze_fusion\yolo_train_output",
        name="power_line_yolo",
        exist_ok=True,
        patience=10,
        save=True,
        amp=False,
        workers=4,
        lr0=0.005,
        verbose=False,
    )
    _save_manifest(manifest)
    print(f"[YOLO] 重训完成, 模型已更新: {YOLO_PATH}")


def load_yolo_model():
    if os.path.exists(YOLO_PATH):
        from ultralytics import YOLO
        return YOLO(YOLO_PATH)
    return None


def detect_with_yolo(yolo_model, img_bgr, conf=0.25):
    if yolo_model is None:
        return []
    results = yolo_model(img_bgr, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            detections.append({
                'class': YOLO_CLASSES[cls_id],
                'cls_id': cls_id,
                'bbox': (x1, y1, x2, y2),
                'conf': conf_val,
            })
    return detections


def draw_yolo_detections(img_bgr, detections):
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = YOLO_COLORS.get(det['class'], (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


# ==================== Mask R-CNN 检测器 ====================
MASKRCNN_CKPT = r"c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth"
MASKRCNN_MEAN = [103.530, 116.280, 123.675]
MASKRCNN_STD = [1.0, 1.0, 1.0]


def _build_maskrcnn_mapping(d2_keys):
    """detectron2 → torchvision 键名映射"""
    res_to_layer = {'res2': 'layer1', 'res3': 'layer2', 'res4': 'layer3', 'res5': 'layer4'}
    mapping = {}
    for d2k in d2_keys:
        tvk = None
        if d2k == 'backbone.bottom_up.stem.conv1.weight':
            tvk = 'backbone.body.conv1.weight'
        elif d2k.startswith('backbone.bottom_up.stem.conv1.norm.'):
            p = d2k.split('norm.')[-1]
            m = {'weight': 'bn1.weight', 'bias': 'bn1.bias', 'running_mean': 'bn1.running_mean', 'running_var': 'bn1.running_var'}
            tvk = f'backbone.body.{m.get(p, "")}' if p in m else None
        elif d2k.startswith('backbone.bottom_up.res'):
            parts = d2k.replace('backbone.bottom_up.', '').split('.')
            ln = res_to_layer.get(parts[0])
            if ln and len(parts) >= 3:
                bid = parts[1]
                rest = '.'.join(parts[2:])
                if rest.startswith('shortcut.'):
                    sp = rest.replace('shortcut.', '')
                    if sp == 'weight': tvk = f'backbone.body.{ln}.{bid}.downsample.0.weight'
                    elif sp.startswith('norm.'):
                        np_p = sp.split('norm.')[-1]
                        m2 = {'weight': '1.weight', 'bias': '1.bias', 'running_mean': '1.running_mean', 'running_var': '1.running_var'}
                        tvk = f'backbone.body.{ln}.{bid}.downsample.{m2.get(np_p, "")}' if np_p in m2 else None
                elif rest.startswith('conv') and '.norm.' in rest:
                    cn = rest.split('.')[0].replace('conv', '')
                    np_p = rest.split('norm.')[-1]
                    m3 = {'weight': f'bn{cn}.weight', 'bias': f'bn{cn}.bias', 'running_mean': f'bn{cn}.running_mean', 'running_var': f'bn{cn}.running_var'}
                    tvk = f'backbone.body.{ln}.{bid}.{m3.get(np_p, "")}' if np_p in m3 else None
                elif rest.startswith('conv') and rest.endswith('weight'):
                    cn = rest.replace('conv', '').replace('.weight', '')
                    tvk = f'backbone.body.{ln}.{bid}.conv{cn}.weight'
        elif d2k.startswith('backbone.fpn_lateral'):
            lv = d2k.split('fpn_lateral')[1][0]
            idx = int(lv) - 2
            s = d2k.split('fpn_lateral' + lv + '.')[-1]
            tvk = f'backbone.fpn.inner_blocks.{idx}.0.{"weight" if s == "weight" else "bias"}'
        elif d2k.startswith('backbone.fpn_output'):
            lv = d2k.split('fpn_output')[1][0]
            idx = int(lv) - 2
            s = d2k.split('fpn_output' + lv + '.')[-1]
            tvk = f'backbone.fpn.layer_blocks.{idx}.0.{"weight" if s == "weight" else "bias"}'
        elif d2k == 'proposal_generator.rpn_head.conv.weight': tvk = 'rpn.head.conv.0.0.weight'
        elif d2k == 'proposal_generator.rpn_head.conv.bias': tvk = 'rpn.head.conv.0.0.bias'
        elif d2k == 'proposal_generator.rpn_head.objectness_logits.weight': tvk = 'rpn.head.cls_logits.weight'
        elif d2k == 'proposal_generator.rpn_head.objectness_logits.bias': tvk = 'rpn.head.cls_logits.bias'
        elif d2k == 'proposal_generator.rpn_head.anchor_deltas.weight': tvk = 'rpn.head.bbox_pred.weight'
        elif d2k == 'proposal_generator.rpn_head.anchor_deltas.bias': tvk = 'rpn.head.bbox_pred.bias'
        elif d2k == 'roi_heads.box_head.fc1.weight': tvk = 'roi_heads.box_head.fc6.weight'
        elif d2k == 'roi_heads.box_head.fc1.bias': tvk = 'roi_heads.box_head.fc6.bias'
        elif d2k == 'roi_heads.box_head.fc2.weight': tvk = 'roi_heads.box_head.fc7.weight'
        elif d2k == 'roi_heads.box_head.fc2.bias': tvk = 'roi_heads.box_head.fc7.bias'
        elif d2k == 'roi_heads.box_predictor.cls_score.weight': tvk = 'roi_heads.box_predictor.cls_score.weight'
        elif d2k == 'roi_heads.box_predictor.cls_score.bias': tvk = 'roi_heads.box_predictor.cls_score.bias'
        elif d2k == 'roi_heads.box_predictor.bbox_pred.weight': tvk = 'roi_heads.box_predictor.bbox_pred.weight'
        elif d2k == 'roi_heads.box_predictor.bbox_pred.bias': tvk = 'roi_heads.box_predictor.bbox_pred.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn1.weight': tvk = 'roi_heads.mask_head.0.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn1.bias': tvk = 'roi_heads.mask_head.0.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn2.weight': tvk = 'roi_heads.mask_head.1.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn2.bias': tvk = 'roi_heads.mask_head.1.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn3.weight': tvk = 'roi_heads.mask_head.2.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn3.bias': tvk = 'roi_heads.mask_head.2.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn4.weight': tvk = 'roi_heads.mask_head.3.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn4.bias': tvk = 'roi_heads.mask_head.3.0.bias'
        elif d2k == 'roi_heads.mask_head.deconv.weight': tvk = 'roi_heads.mask_predictor.conv5_mask.weight'
        elif d2k == 'roi_heads.mask_head.deconv.bias': tvk = 'roi_heads.mask_predictor.conv5_mask.bias'
        elif d2k == 'roi_heads.mask_head.predictor.weight': tvk = 'roi_heads.mask_predictor.mask_fcn_logits.weight'
        elif d2k == 'roi_heads.mask_head.predictor.bias': tvk = 'roi_heads.mask_predictor.mask_fcn_logits.bias'
        if tvk:
            mapping[d2k] = tvk
    return mapping


def load_maskrcnn_model():
    """加载detectron2 Mask R-CNN权重到torchvision模型"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    ckpt = torch.load(MASKRCNN_CKPT, map_location='cpu', weights_only=False)
    d2_sd = ckpt['model']
    model = maskrcnn_resnet50_fpn(
        weights=None, num_classes=2, weights_backbone=None,
        min_size=800, max_size=1333,
        box_score_thresh=0.05, box_nms_thresh=0.3,
        box_detections_per_img=20)
    model.transform.image_mean = MASKRCNN_MEAN
    model.transform.image_std = MASKRCNN_STD

    tv_sd = model.state_dict()
    mapping = _build_maskrcnn_mapping(list(d2_sd.keys()))
    new_sd = {}
    for d2k, tvk in mapping.items():
        if tvk in tv_sd:
            d2v, tvv = d2_sd[d2k], tv_sd[tvk]
            if d2v.shape == tvv.shape:
                new_sd[tvk] = d2v.clone()
            elif 'bbox_pred' in tvk or 'mask_fcn_logits' in tvk:
                new_sd[tvk] = torch.cat([d2v, d2v], dim=0)
    for tvk in tv_sd:
        if tvk not in new_sd and 'num_batches_tracked' in tvk:
            new_sd[tvk] = torch.tensor(0, dtype=torch.long)
    model.load_state_dict(new_sd, strict=False)
    model.to(DEVICE).eval()
    return model


def detect_with_maskrcnn(model, img_bgr, class_name='target', min_area=200, iou_thresh=0.3):
    """Mask R-CNN推理, 返回与YOLO兼容的检测格式 + 实例掩码"""
    img_t = torch.from_numpy(img_bgr).permute(2, 0, 1).float().to(DEVICE)
    with torch.no_grad():
        outputs = model([img_t])
    pred = outputs[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    masks = pred['masks'].cpu().numpy() if 'masks' in pred else None

    # 过滤小框
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = areas >= min_area
    boxes, scores = boxes[keep], scores[keep]
    if masks is not None:
        masks = masks[keep]

    # 额外NMS去重
    if len(boxes) > 1:
        keep_idx = tv_nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thresh).numpy()
        boxes, scores = boxes[keep_idx], scores[keep_idx]
        if masks is not None:
            masks = masks[keep_idx]

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        det = {'class': class_name, 'cls_id': 1, 'bbox': (x1, y1, x2, y2), 'conf': float(scores[i])}
        if masks is not None:
            det['mask'] = masks[i, 0]
        detections.append(det)
    return detections


def draw_maskrcnn_detections(img_bgr, detections):
    """绘制Mask R-CNN检测结果 (含实例掩码)"""
    annotated = img_bgr.copy()
    color = (0, 0, 255)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        if 'mask' in det:
            mask_bool = det['mask'] > 0.5
            colored = np.zeros_like(annotated)
            colored[:] = color
            annotated = np.where(mask_bool[:, :, None],
                                cv2.addWeighted(annotated, 0.7, colored, 0.3, 0), annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def generate_ice_mask(img_bgr, yolo_detections=None):
    """输电线路覆冰检测

    改进逻辑:
    1. 优先用YOLO检测的power_line/insulator区域作为走廊
    2. 覆冰需同时满足: 白色(低饱和高亮度) 且 中等纹理(排除水印锐边)
    3. 轮廓形状过滤: 排除极端长宽比和过小区域
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # Step 1: 颜色筛选 — 低饱和度 + 较高亮度 (覆冰白色特征)
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, low_sat = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v_mean = np.mean(v)
    _, bright = cv2.threshold(v, int(v_mean * 0.7), 255, cv2.THRESH_BINARY)
    ice_color = cv2.bitwise_and(low_sat, bright)

    # Step 2: 构建走廊 (YOLO power_line/insulator 区域, 或 Hough 直线)
    corridor = np.zeros((h, w), dtype=np.uint8)
    use_yolo = yolo_detections is not None and len(yolo_detections) > 0

    if use_yolo:
        for det in yolo_detections:
            if det['class'] in ('power_line', 'insulator'):
                x1, y1, x2, y2 = det['bbox']
                pad = 15
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                cv2.rectangle(corridor, (x1, y1), (x2, y2), 255, -1)
    else:
        edges = cv2.Canny(gray, 20, 80)
        min_len = max(50, min(h, w) // 4)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                               minLineLength=min_len, maxLineGap=20)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                pts = line.reshape(-1) if line.ndim > 1 else line
                if len(pts) >= 4:
                    x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length >= min_len:
                        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 20)
        corridor = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        corridor_ratio = np.sum(corridor > 0) / (h * w)
        if corridor_ratio > 0.4:
            corridor = np.zeros((h, w), dtype=np.uint8)

    # Step 3: 覆冰纹理 — 中等梯度 (排除极高地梯度的水印文字锐边)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = (grad_mag / (grad_mag.max() + 1e-6) * 255).astype(np.uint8)
    grad_blur = cv2.GaussianBlur(grad_norm, (5, 5), 0)
    _, texture_mask = cv2.threshold(grad_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 排除极高梯度区域 (水印文字锐边, 远高于覆冰表面纹理)
    _, high_grad = cv2.threshold(grad_blur, 180, 255, cv2.THRESH_BINARY)
    texture_mask = cv2.subtract(texture_mask, high_grad)

    # Step 4: 组合 — 走廊内 且 白色 且 有中等纹理 (AND, 非OR)
    has_corridor = np.sum(corridor) > 0
    if has_corridor:
        combined = cv2.bitwise_and(cv2.bitwise_and(ice_color, texture_mask), corridor)
        if np.sum(combined) < 100:
            combined = cv2.bitwise_and(ice_color, corridor)
    else:
        combined = np.zeros((h, w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Step 5: 轮廓形状过滤 — 排除极端长宽比和过小区域
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros((h, w), dtype=np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect > 8.0 or aspect < 0.125:
            continue
        cv2.drawContours(final, [c], -1, 255, -1)

    return final


def _filter_ice_detections(detections):
    """过滤YOLO ice检测结果: 仅保留与power_line/insulator重叠的ice检测"""
    line_boxes = [d for d in detections if d['class'] in ('power_line', 'insulator')]
    if not line_boxes:
        return [d for d in detections if d['class'] != 'ice']

    def overlaps(det, boxes):
        dx1, dy1, dx2, dy2 = det['bbox']
        for b in boxes:
            bx1, by1, bx2, by2 = b['bbox']
            ix1 = max(dx1, bx1)
            iy1 = max(dy1, by1)
            ix2 = min(dx2, bx2)
            iy2 = min(dy2, by2)
            if ix1 < ix2 and iy1 < iy2:
                return True
        return False

    filtered = []
    for det in detections:
        if det['class'] == 'ice':
            if overlaps(det, line_boxes):
                filtered.append(det)
        else:
            filtered.append(det)
    return filtered


def process(model, name, input_path, output_path, yolo_model=None,
            ice_mask=False, compare=False, input_dir=None,
            maskrcnn_model=None):
    img = cv2.imread(input_path)
    if img is None:
        print(f"  [跳过] 无法读取: {input_path}")
        return None

    result = dehaze_image(model, img, name)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result)

    use_maskrcnn = maskrcnn_model is not None
    if use_maskrcnn:
        yolo_dets = detect_with_maskrcnn(maskrcnn_model, result)
    else:
        yolo_dets = detect_with_yolo(yolo_model, result) if yolo_model else []
        yolo_dets = _filter_ice_detections(yolo_dets)

    if yolo_dets:
        if use_maskrcnn:
            annotated = draw_maskrcnn_detections(result, yolo_dets)
            base, ext = os.path.splitext(output_path)
            det_path = f"{base}_maskrcnn{ext}"
        else:
            annotated = draw_yolo_detections(result, yolo_dets)
            base, ext = os.path.splitext(output_path)
            det_path = f"{base}_yolo{ext}"
        cv2.imwrite(det_path, annotated)

    mask = None
    ice_ratio = 0.0
    has_ice = False
    if ice_mask:
        mask = generate_ice_mask(result, yolo_dets)
        ice_ratio = np.sum(mask > 127) / mask.size * 100
        if ice_ratio > 0.01:
            has_ice = True
            base, ext = os.path.splitext(output_path)
            mask_path = f"{base}_ice_mask{ext}"
            cv2.imwrite(mask_path, mask)

        n_dets = len(yolo_dets)
        det_summary = ", ".join(f"{d['class']}({d['conf']:.2f})" for d in yolo_dets) if yolo_dets else "无"
        ice_str = f"覆冰: {ice_ratio:.1f}%" if has_ice else "无覆冰"
        det_name = "MaskRCNN" if use_maskrcnn else "YOLO"
        print(f"  [完成] {os.path.basename(input_path)} | {det_name}: {n_dets}个检测 [{det_summary}] | {ice_str}")
    else:
        n_dets = len(yolo_dets)
        det_summary = ", ".join(f"{d['class']}({d['conf']:.2f})" for d in yolo_dets) if yolo_dets else "无"
        det_name = "MaskRCNN" if use_maskrcnn else "YOLO"
        print(f"  [完成] {os.path.basename(input_path)} | {det_name}: {n_dets}个检测 [{det_summary}]")

    if compare:
        effective_mask = mask if has_ice else None
        generate_comparison(input_path, output_path, effective_mask, yolo_dets, output_path, input_dir)

    return result


def generate_comparison(orig_path, dehazed_path, ice_mask, yolo_dets, ref_output_path, input_dir):
    orig = cv2.imread(orig_path)
    dehazed = cv2.imread(dehazed_path)
    if orig is None or dehazed is None:
        return

    h = max(orig.shape[0], dehazed.shape[0])
    w = max(orig.shape[1], dehazed.shape[1])

    def resize_pad(img, target_h, target_w):
        ih, iw = img.shape[:2]
        scale = min(target_h / ih, target_w / iw)
        nh, nw = int(ih * scale), int(iw * scale)
        img = cv2.resize(img, (nw, nh))
        pad_h = target_h - nh
        pad_w = target_w - nw
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        return img

    orig_r = resize_pad(orig, h, w)
    dehazed_r = resize_pad(dehazed, h, w)

    yolo_annotated = dehazed_r.copy()
    has_yolo = yolo_dets is not None and len(yolo_dets) > 0
    has_ice = ice_mask is not None
    is_maskrcnn = has_yolo and any('mask' in d for d in yolo_dets)
    det_label = "Mask R-CNN" if is_maskrcnn else "YOLO"

    def draw_dets(canvas, dets, scale_x, scale_y):
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            color = YOLO_COLORS.get(det['class'], (0, 0, 255))
            if 'mask' in det:
                mask_resized = cv2.resize(det['mask'], (dehazed_r.shape[1], dehazed_r.shape[0]))
                mask_bool = mask_resized > 0.5
                colored_layer = np.zeros_like(canvas)
                colored_layer[:] = color
                canvas = np.where(mask_bool[:, :, None],
                                  cv2.addWeighted(canvas, 0.7, colored_layer, 0.3, 0), canvas)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(canvas, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    if has_yolo:
        scale_x = dehazed_r.shape[1] / dehazed.shape[1]
        scale_y = dehazed_r.shape[0] / dehazed.shape[0]
        yolo_annotated = draw_dets(yolo_annotated, yolo_dets, scale_x, scale_y)

    if has_ice:
        mask_bool = cv2.resize(ice_mask, (dehazed_r.shape[1], dehazed_r.shape[0])) > 127
        red_layer = np.zeros_like(dehazed_r)
        red_layer[:] = [0, 0, 255]
        ice_overlay = np.where(mask_bool[:, :, None],
                               cv2.addWeighted(dehazed_r, 0.6, red_layer, 0.4, 0), dehazed_r)
    else:
        ice_overlay = yolo_annotated if has_yolo else dehazed_r

    if has_yolo and has_ice:
        ice_overlay_with_yolo = draw_dets(ice_overlay.copy(), yolo_dets, scale_x, scale_y)
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r, gap, yolo_annotated, gap, ice_overlay_with_yolo])
        labels = ["Original", "Dehazed", f"{det_label} Detection", f"Ice + {det_label}"]
    elif has_yolo:
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r, gap, yolo_annotated])
        labels = ["Original", "Dehazed", f"{det_label} Detection"]
    elif has_ice:
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r, gap, ice_overlay])
        labels = ["Original", "Dehazed", "Ice Detection (Red)"]
    else:
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r])
        labels = ["Original", "Dehazed"]

    label_h = 30
    label_bar = np.full((label_h, combined.shape[1], 3), 240, dtype=np.uint8)
    n_panels = len(labels)
    panel_w = w
    for i, label in enumerate(labels):
        cx = i * (panel_w + 10) + panel_w // 2
        cv2.putText(label_bar, label, (cx - len(label) * 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    final = np.vstack([label_bar, combined])

    compare_dir = os.path.normpath(os.path.join(os.path.dirname(ref_output_path), "..", "compare"))
    os.makedirs(compare_dir, exist_ok=True)

    fname = os.path.basename(orig_path)
    out_path = os.path.join(compare_dir, f"compare_{fname}")
    cv2.imwrite(out_path, final)


def main():
    parser = argparse.ArgumentParser(description="IceWave-DehazeFormer 图片去雾 + YOLO检测")
    parser.add_argument("--input", "-i", required=True, help="输入图片路径或文件夹")
    parser.add_argument("--output", "-o", required=True, help="输出图片路径或文件夹")
    parser.add_argument("--model", "-m", default="m4", choices=list(CKPT.keys()),
                        help="模型选择 (默认m4, 性能最佳)")
    parser.add_argument("--ice-mask", action="store_true",
                        help="同时生成覆冰区域掩码")
    parser.add_argument("--compare", action="store_true",
                        help="生成原图 vs 去雾 vs YOLO vs 覆冰检测的并排对比图")
    parser.add_argument("--yolo", action="store_true", default=True,
                        help="启用YOLO检测 (默认开启, 存在训练模型时自动加载)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="禁用YOLO检测")
    parser.add_argument("--no-retrain", action="store_true",
                        help="跳过YOLO自动重训 (仅用现有模型推理)")
    parser.add_argument("--use-maskrcnn", action="store_true",
                        help="使用同事提供的Mask R-CNN模型替代YOLO进行目标检测")
    args = parser.parse_args()

    use_yolo = args.yolo and not args.no_yolo
    use_maskrcnn = args.use_maskrcnn

    print(f"设备: {DEVICE}")
    if DEVICE == "cpu":
        print("警告: 未检测到GPU, 将使用CPU推理 (速度较慢)")

    if use_yolo and not use_maskrcnn and not args.no_retrain and os.path.isdir(args.input):
        check_and_retrain_yolo(args.input)

    print(f"加载去雾模型: {args.model} ({MODEL_DESC[args.model]})...")
    model = load_model(args.model)
    print(f"去雾模型加载完成")

    yolo_model = None
    maskrcnn_model = None

    if use_maskrcnn:
        print(f"加载Mask R-CNN模型 (同事提供)...")
        maskrcnn_model = load_maskrcnn_model()
        print(f"Mask R-CNN模型加载完成: {MASKRCNN_CKPT}")
    elif use_yolo:
        print(f"加载YOLO模型...")
        yolo_model = load_yolo_model()
        if yolo_model:
            print(f"YOLO模型加载完成: {YOLO_PATH}")
        else:
            print(f"未找到YOLO模型 (训练未完成), 跳过YOLO检测")
    print()

    if os.path.isfile(args.input):
        process(model, args.model, args.input, args.output, yolo_model,
                args.ice_mask, args.compare, maskrcnn_model=maskrcnn_model)
    elif os.path.isdir(args.input):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(exts)])
        if not files:
            print(f"错误: 文件夹中未找到图片文件")
            return
        print(f"找到 {len(files)} 张图片\n")
        os.makedirs(args.output, exist_ok=True)
        for i, fname in enumerate(files):
            inp = os.path.join(args.input, fname)
            out = os.path.join(args.output, fname)
            process(model, args.model, inp, out, yolo_model,
                    args.ice_mask, args.compare, args.input,
                    maskrcnn_model=maskrcnn_model)
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(files)}")
        print(f"\n全部完成: {len(files)} 张图片已处理 → {args.output}")
        if args.compare:
            compare_dir = os.path.normpath(os.path.join(args.output, "..", "compare"))
            print(f"对比图已保存 → {compare_dir}")
    else:
        print(f"错误: 路径不存在: {args.input}")


if __name__ == "__main__":
    main()
