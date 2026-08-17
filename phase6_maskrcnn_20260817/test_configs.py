"""优化Mask R-CNN推理参数并可视化检测结果"""
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d")

from maskrcnn_inference import build_mapping, CKPT_PATH, DEVICE, D2_MEAN, D2_STD
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import nms


def load_model(detections_per_img=20, score_thresh=0.05, nms_thresh=0.3):
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    d2_sd = ckpt['model']
    model = maskrcnn_resnet50_fpn(
        weights=None, num_classes=2, weights_backbone=None,
        min_size=800, max_size=1333,
        box_score_thresh=score_thresh, box_nms_thresh=nms_thresh,
        box_detections_per_img=detections_per_img)
    model.transform.image_mean = D2_MEAN
    model.transform.image_std = D2_STD

    tv_sd = model.state_dict()
    mapping = build_mapping(list(d2_sd.keys()), list(tv_sd.keys()))
    new_sd = {}
    for d2k, tvk in mapping.items():
        if tvk in tv_sd:
            d2_val = d2_sd[d2k]
            tv_val = tv_sd[tvk]
            if d2_val.shape == tv_val.shape:
                new_sd[tvk] = d2_val.clone()
            elif 'bbox_pred' in tvk:
                new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
            elif 'mask_fcn_logits' in tvk:
                new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
    for tvk in tv_sd:
        if tvk not in new_sd and 'num_batches_tracked' in tvk:
            new_sd[tvk] = torch.tensor(0, dtype=torch.long)
    model.load_state_dict(new_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_with_filter(model, img_bgr, min_box_area=100, iou_thresh=0.3):
    """推理 + 后处理过滤"""
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float().to(DEVICE)
    with torch.no_grad():
        outputs = model([img_tensor])
    pred = outputs[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    masks = pred['masks'].cpu().numpy() if 'masks' in pred else None

    # 过滤小框
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = areas >= min_box_area
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if masks is not None:
        masks = masks[keep]

    # 额外NMS
    if len(boxes) > 1:
        keep_idx = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thresh).numpy()
        boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
        if masks is not None:
            masks = masks[keep_idx]

    return {'boxes': boxes, 'scores': scores, 'labels': labels, 'masks': masks}


def draw_results(img_bgr, result, class_name='target'):
    """绘制检测框和掩码"""
    annotated = img_bgr.copy()
    boxes = result['boxes']
    scores = result['scores']
    masks = result.get('masks')

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = (0, 0, 255)

        if masks is not None and i < len(masks):
            mask = masks[i, 0]
            mask_bool = mask > 0.5
            colored = np.zeros_like(annotated)
            colored[:] = color
            annotated = np.where(mask_bool[:, :, None],
                                cv2.addWeighted(annotated, 0.7, colored, 0.3, 0),
                                annotated)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def make_comparison(orig, annotated, mask_overlay=None):
    """生成对比图"""
    h = max(orig.shape[0], annotated.shape[0])
    w = max(orig.shape[1], annotated.shape[1])

    def resize_pad(img, th, tw):
        ih, iw = img.shape[:2]
        s = min(th/ih, tw/iw)
        nh, nw = int(ih*s), int(iw*s)
        img = cv2.resize(img, (nw, nh))
        ph, pw = th-nh, tw-nw
        if ph > 0 or pw > 0:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=[128,128,128])
        return img

    panels = [resize_pad(orig, h, w), resize_pad(annotated, h, w)]
    labels = ["Original", "Mask R-CNN Detection"]
    if mask_overlay is not None:
        panels.append(resize_pad(mask_overlay, h, w))
        labels.append("Mask Overlay")

    gap = np.full((h, 10, 3), 255, dtype=np.uint8)
    combined = np.hstack([panels[0]] + [gap] + [panels[1]])
    if len(panels) > 2:
        combined = np.hstack([combined, gap, panels[2]])

    bar = np.full((30, combined.shape[1], 3), 240, dtype=np.uint8)
    pw = w
    for i, label in enumerate(labels):
        cx = i * (pw + 10) + pw // 2
        cv2.putText(bar, label, (cx - len(label)*4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    return np.vstack([bar, combined])


if __name__ == "__main__":
    model = load_model(detections_per_img=20, score_thresh=0.05, nms_thresh=0.3)

    input_dir = r"D:\dehaze_fusion\my_test\input"
    output_dir = r"D:\dehaze_fusion\my_test\output_maskrcnn"
    compare_dir = r"D:\dehaze_fusion\my_test\compare_maskrcnn"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.JPG'))])
    print(f"处理 {len(files)} 张图片...")

    for fname in files:
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue

        result = predict_with_filter(model, img, min_box_area=200, iou_thresh=0.3)
        n = len(result['boxes'])

        annotated = draw_results(img, result)

        # 掩码叠加
        mask_overlay = None
        if result.get('masks') is not None and len(result['masks']) > 0:
            mask_overlay = img.copy()
            total_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for i in range(len(result['masks'])):
                m = result['masks'][i, 0] > 0.5
                total_mask = total_mask | m.astype(np.uint8) * 255
            red = np.zeros_like(mask_overlay)
            red[:] = [0, 0, 255]
            mask_overlay = np.where(total_mask[:, :, None] > 0,
                                    cv2.addWeighted(mask_overlay, 0.6, red, 0.4, 0),
                                    mask_overlay)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, annotated)

        if mask_overlay is not None:
            base, ext = os.path.splitext(out_path)
            cv2.imwrite(f"{base}_mask{ext}", mask_overlay)
            ice_ratio = np.sum(total_mask > 127) / total_mask.size * 100
            print(f"  {fname}: {n}个检测, 掩码面积 {ice_ratio:.1f}%")
        else:
            print(f"  {fname}: {n}个检测, 无掩码")

        # 对比图
        comp = make_comparison(img, annotated, mask_overlay)
        cv2.imwrite(os.path.join(compare_dir, f"compare_{fname}"), comp)

    print(f"\n输出: {output_dir}")
    print(f"对比图: {compare_dir}")
