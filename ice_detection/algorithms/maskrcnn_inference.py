"""
加载同事提供的detectron2 Mask R-CNN权重到torchvision模型进行推理

模型: Mask R-CNN R50-FPN, detectron2训练, 2类(背景+1前景), 1掩码通道
用途: 替代YOLO进行更精准的目标检测
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform

CKPT_PATH = r"c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# detectron2 BGR归一化 (pixel_mean=[103.530,116.280,123.675], pixel_std=[1.0,1.0,1.0])
# torchvision transform对float输入不会自动除255, 所以mean/std保持[0,255]范围
D2_MEAN = [103.530, 116.280, 123.675]  # BGR, [0,255]范围
D2_STD = [1.0, 1.0, 1.0]


def build_mapping(d2_keys, tv_keys):
    """构建detectron2→torchvision的键名映射"""
    mapping = {}

    res_to_layer = {'res2': 'layer1', 'res3': 'layer2', 'res4': 'layer3', 'res5': 'layer4'}

    for d2k in d2_keys:
        tvk = None

        # === Backbone: stem ===
        if d2k == 'backbone.bottom_up.stem.conv1.weight':
            tvk = 'backbone.body.conv1.weight'
        elif d2k.startswith('backbone.bottom_up.stem.conv1.norm.'):
            norm_part = d2k.split('norm.')[-1]
            if norm_part == 'running_mean':
                tvk = 'backbone.body.bn1.running_mean'
            elif norm_part == 'running_var':
                tvk = 'backbone.body.bn1.running_var'
            elif norm_part == 'weight':
                tvk = 'backbone.body.bn1.weight'
            elif norm_part == 'bias':
                tvk = 'backbone.body.bn1.bias'

        # === Backbone: res blocks ===
        elif d2k.startswith('backbone.bottom_up.res'):
            parts = d2k.replace('backbone.bottom_up.', '').split('.')
            res_name = parts[0]  # res2, res3, res4, res5
            layer_name = res_to_layer.get(res_name)
            if layer_name and len(parts) >= 3:
                block_id = parts[1]
                rest = '.'.join(parts[2:])

                if rest.startswith('shortcut.'):
                    sc_part = rest.replace('shortcut.', '')
                    if sc_part == 'weight':
                        tvk = f'backbone.body.{layer_name}.{block_id}.downsample.0.weight'
                    elif sc_part.startswith('norm.'):
                        np_part = sc_part.split('norm.')[-1]
                        if np_part == 'weight':
                            tvk = f'backbone.body.{layer_name}.{block_id}.downsample.1.weight'
                        elif np_part == 'bias':
                            tvk = f'backbone.body.{layer_name}.{block_id}.downsample.1.bias'
                        elif np_part == 'running_mean':
                            tvk = f'backbone.body.{layer_name}.{block_id}.downsample.1.running_mean'
                        elif np_part == 'running_var':
                            tvk = f'backbone.body.{layer_name}.{block_id}.downsample.1.running_var'

                elif rest.startswith('conv') and '.norm.' in rest:
                    conv_num = rest.split('.')[0].replace('conv', '')  # 1, 2, 3
                    norm_part = rest.split('norm.')[-1]
                    bn_map = {'weight': f'bn{conv_num}.weight', 'bias': f'bn{conv_num}.bias',
                              'running_mean': f'bn{conv_num}.running_mean',
                              'running_var': f'bn{conv_num}.running_var'}
                    if norm_part in bn_map:
                        tvk = f'backbone.body.{layer_name}.{block_id}.{bn_map[norm_part]}'

                elif rest.startswith('conv') and rest.endswith('weight'):
                    conv_num = rest.replace('conv', '').replace('.weight', '')
                    tvk = f'backbone.body.{layer_name}.{block_id}.conv{conv_num}.weight'

        # === FPN ===
        elif d2k.startswith('backbone.fpn_lateral'):
            level = d2k.split('fpn_lateral')[1][0]  # 2,3,4,5
            inner_idx = int(level) - 2  # 2→0, 3→1, 4→2, 5→3
            suffix = d2k.split('fpn_lateral' + level + '.')[-1]
            if suffix == 'weight':
                tvk = f'backbone.fpn.inner_blocks.{inner_idx}.0.weight'
            elif suffix == 'bias':
                tvk = f'backbone.fpn.inner_blocks.{inner_idx}.0.bias'

        elif d2k.startswith('backbone.fpn_output'):
            level = d2k.split('fpn_output')[1][0]
            layer_idx = int(level) - 2
            suffix = d2k.split('fpn_output' + level + '.')[-1]
            if suffix == 'weight':
                tvk = f'backbone.fpn.layer_blocks.{layer_idx}.0.weight'
            elif suffix == 'bias':
                tvk = f'backbone.fpn.layer_blocks.{layer_idx}.0.bias'

        # === RPN ===
        elif d2k == 'proposal_generator.rpn_head.conv.weight':
            tvk = 'rpn.head.conv.0.0.weight'
        elif d2k == 'proposal_generator.rpn_head.conv.bias':
            tvk = 'rpn.head.conv.0.0.bias'
        elif d2k == 'proposal_generator.rpn_head.objectness_logits.weight':
            tvk = 'rpn.head.cls_logits.weight'
        elif d2k == 'proposal_generator.rpn_head.objectness_logits.bias':
            tvk = 'rpn.head.cls_logits.bias'
        elif d2k == 'proposal_generator.rpn_head.anchor_deltas.weight':
            tvk = 'rpn.head.bbox_pred.weight'
        elif d2k == 'proposal_generator.rpn_head.anchor_deltas.bias':
            tvk = 'rpn.head.bbox_pred.bias'

        # === Box head ===
        elif d2k == 'roi_heads.box_head.fc1.weight':
            tvk = 'roi_heads.box_head.fc6.weight'
        elif d2k == 'roi_heads.box_head.fc1.bias':
            tvk = 'roi_heads.box_head.fc6.bias'
        elif d2k == 'roi_heads.box_head.fc2.weight':
            tvk = 'roi_heads.box_head.fc7.weight'
        elif d2k == 'roi_heads.box_head.fc2.bias':
            tvk = 'roi_heads.box_head.fc7.bias'

        # === Box predictor ===
        elif d2k == 'roi_heads.box_predictor.cls_score.weight':
            tvk = 'roi_heads.box_predictor.cls_score.weight'
        elif d2k == 'roi_heads.box_predictor.cls_score.bias':
            tvk = 'roi_heads.box_predictor.cls_score.bias'
        elif d2k == 'roi_heads.box_predictor.bbox_pred.weight':
            tvk = 'roi_heads.box_predictor.bbox_pred.weight'
        elif d2k == 'roi_heads.box_predictor.bbox_pred.bias':
            tvk = 'roi_heads.box_predictor.bbox_pred.bias'

        # === Mask head ===
        elif d2k == 'roi_heads.mask_head.mask_fcn1.weight':
            tvk = 'roi_heads.mask_head.0.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn1.bias':
            tvk = 'roi_heads.mask_head.0.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn2.weight':
            tvk = 'roi_heads.mask_head.1.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn2.bias':
            tvk = 'roi_heads.mask_head.1.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn3.weight':
            tvk = 'roi_heads.mask_head.2.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn3.bias':
            tvk = 'roi_heads.mask_head.2.0.bias'
        elif d2k == 'roi_heads.mask_head.mask_fcn4.weight':
            tvk = 'roi_heads.mask_head.3.0.weight'
        elif d2k == 'roi_heads.mask_head.mask_fcn4.bias':
            tvk = 'roi_heads.mask_head.3.0.bias'
        elif d2k == 'roi_heads.mask_head.deconv.weight':
            tvk = 'roi_heads.mask_predictor.conv5_mask.weight'
        elif d2k == 'roi_heads.mask_head.deconv.bias':
            tvk = 'roi_heads.mask_predictor.conv5_mask.bias'
        elif d2k == 'roi_heads.mask_head.predictor.weight':
            tvk = 'roi_heads.mask_predictor.mask_fcn_logits.weight'
        elif d2k == 'roi_heads.mask_head.predictor.bias':
            tvk = 'roi_heads.mask_predictor.mask_fcn_logits.bias'

        if tvk:
            mapping[d2k] = tvk

    return mapping


def load_detectron2_model():
    """加载detectron2权重到torchvision Mask R-CNN模型"""
    print("加载detectron2检查点...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    d2_sd = ckpt['model']

    print("创建torchvision Mask R-CNN R50-FPN (num_classes=2)...")
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=2, weights_backbone=None,
                                  min_size=800, max_size=1333,
                                  box_score_thresh=0.5, box_nms_thresh=0.5,
                                  box_detections_per_img=50)

    # 修改transform以匹配detectron2的BGR归一化
    model.transform.image_mean = D2_MEAN
    model.transform.image_std = D2_STD

    tv_sd = model.state_dict()
    mapping = build_mapping(list(d2_sd.keys()), list(tv_sd.keys()))

    print(f"键名映射: {len(mapping)} / {len(d2_sd)} detectron2参数")

    # 构建新的state_dict
    new_sd = {}
    matched = 0
    shape_mismatch = []

    for d2k, tvk in mapping.items():
        if tvk in tv_sd:
            d2_val = d2_sd[d2k]
            tv_val = tv_sd[tvk]

            if d2_val.shape == tv_val.shape:
                new_sd[tvk] = d2_val
                matched += 1
            else:
                # 处理形状不匹配
                if tvk == 'roi_heads.box_predictor.bbox_pred.weight':
                    # detectron2 [4, 1024] → torchvision [8, 1024] (复制为2类)
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                    matched += 1
                elif tvk == 'roi_heads.box_predictor.bbox_pred.bias':
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                    matched += 1
                elif tvk == 'roi_heads.mask_predictor.mask_fcn_logits.weight':
                    # detectron2 [1, 256, 1, 1] → torchvision [2, 256, 1, 1]
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                    matched += 1
                elif tvk == 'roi_heads.mask_predictor.mask_fcn_logits.bias':
                    new_sd[tvk] = torch.cat([d2_val, d2_val], dim=0)
                    matched += 1
                else:
                    shape_mismatch.append((d2k, tvk, d2_val.shape, tv_val.shape))

    # 保留num_batches_tracked等未映射的参数
    for tvk in tv_sd:
        if tvk not in new_sd and 'num_batches_tracked' in tvk:
            new_sd[tvk] = torch.tensor(0, dtype=torch.long)

    print(f"成功映射: {matched} / {len(tv_sd)} torchvision参数")
    if shape_mismatch:
        print(f"形状不匹配(未处理): {len(shape_mismatch)}")
        for d2k, tvk, s1, s2 in shape_mismatch[:5]:
            print(f"  {d2k}({s1}) → {tvk}({s2})")

    # 统计未映射的torchvision参数
    unmapped = [k for k in tv_sd if k not in new_sd]
    if unmapped:
        print(f"未映射的torchvision参数: {len(unmapped)}")
        for k in unmapped[:10]:
            print(f"  {k}: {tv_sd[k].shape}")

    # 加载权重
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"缺失参数: {len(missing)} (前10: {missing[:10]})")
    if unexpected:
        print(f"多余参数: {len(unexpected)} (前10: {unexpected[:10]})")

    model.to(DEVICE)
    model.eval()
    print("模型加载完成!")
    return model


def predict(model, img_bgr, conf_thresh=0.7):
    """对BGR图像进行推理"""
    h, w = img_bgr.shape[:2]
    # cv2读取的是BGR, detectron2也用BGR, 直接用
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float().to(DEVICE)

    with torch.no_grad():
        outputs = model([img_tensor])

    pred = outputs[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    masks = pred['masks'].cpu().numpy() if 'masks' in pred else None

    # 过滤低置信度
    keep = scores >= conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if masks is not None:
        masks = masks[keep]

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'masks': masks,
    }


def draw_predictions(img_bgr, result, class_name='target'):
    """在图像上绘制检测框和掩码"""
    annotated = img_bgr.copy()
    boxes = result['boxes']
    scores = result['scores']
    masks = result.get('masks')

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255)]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = colors[i % len(colors)]

        # 绘制掩码
        if masks is not None and i < len(masks):
            mask = masks[i, 0]  # [H, W] probability
            mask_bool = mask > 0.5
            colored = np.zeros_like(annotated)
            colored[:] = color
            annotated = np.where(mask_bool[:, :, None],
                                cv2.addWeighted(annotated, 0.6, colored, 0.4, 0),
                                annotated)

        # 绘制框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def generate_mask_overlay(img_bgr, result):
    """生成掩码叠加图"""
    overlay = img_bgr.copy()
    masks = result.get('masks')
    if masks is None or len(masks) == 0:
        return None

    total_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for i in range(len(masks)):
        mask = masks[i, 0] > 0.5
        total_mask = total_mask | mask.astype(np.uint8) * 255

    red_layer = np.zeros_like(overlay)
    red_layer[:] = [0, 0, 255]
    overlay = np.where(total_mask[:, :, None] > 0,
                       cv2.addWeighted(overlay, 0.6, red_layer, 0.4, 0),
                       overlay)
    return overlay, total_mask


if __name__ == "__main__":
    model = load_detectron2_model()

    test_dir = r"D:\dehaze_fusion\my_test\input"
    output_dir = r"D:\dehaze_fusion\my_test\output_maskrcnn"
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG')
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(exts)])
    print(f"\n处理 {len(files)} 张图片...")

    for fname in files:
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = predict(model, img, conf_thresh=0.5)
        n_det = len(result['boxes'])

        annotated = draw_predictions(img, result)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, annotated)

        mask_result = generate_mask_overlay(img, result)
        if mask_result is not None:
            overlay, mask = mask_result
            base, ext = os.path.splitext(out_path)
            cv2.imwrite(f"{base}_mask{ext}", mask)
            ice_ratio = np.sum(mask > 127) / mask.size * 100
            print(f"  {fname}: {n_det}个检测, 掩码面积 {ice_ratio:.1f}%")
        else:
            print(f"  {fname}: {n_det}个检测, 无掩码")
