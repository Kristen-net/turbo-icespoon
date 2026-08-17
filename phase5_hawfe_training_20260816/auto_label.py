"""
自动标注脚本 - 用OpenCV检测输电线路组件
生成YOLO格式标注文件 (.txt)

类别:
  0: insulator  (绝缘子 - 通过颜色/纹理检测)
  1: power_line (电力线 - 通过Hough直线检测)
  2: ice        (覆冰 - 通过白色+纹理检测)
  3: tower      (铁塔 - 通过垂直边缘结构检测)
"""
import os
import cv2
import numpy as np
import glob
import random

# ==================== 配置 ====================
INPUT_DIR = r"D:\DATA_ALL\dataset\test\hazy_real"
OUTPUT_DIR = r"D:\dehaze_fusion\yolo_dataset\images"
LABEL_DIR = r"D:\dehaze_fusion\yolo_dataset\labels"
TRAIN_RATIO = 0.8  # 80%训练, 20%验证

CLASSES = ['insulator', 'power_line', 'ice', 'tower']

IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG')


def detect_insulators(img):
    """检测绝缘子 - 基于颜色分割（棕色/灰色/陶瓷色）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    boxes = []
    
    # 绝缘子通常是棕色/暗红色 (陶瓷绝缘子) 或灰色/白色 (复合绝缘子)
    # 检测棕色范围
    mask1 = cv2.inRange(hsv, np.array([5, 30, 40]), np.array([25, 200, 200]))
    # 检测灰色范围 (复合绝缘子)
    mask2 = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 180]))
    
    mask = mask1 | mask2
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # 太小，跳过
            continue
        if area > h * w * 0.3:  # 太大，可能是背景
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        
        # 绝缘子通常是竖长形 (aspect < 0.8) 或圆形
        if aspect < 2.0 and bh > 20:
            boxes.append((0, x, y, bw, bh))  # class 0: insulator
    
    return boxes


def detect_power_lines(img):
    """检测电力线 - 基于Hough直线变换"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=min(h, w) // 4,
                           maxLineGap=30)
    
    boxes = []
    if lines is not None:
        # 聚合相近的直线
        line_groups = []
        for line in lines:
            line_arr = line[0] if len(line.shape) > 1 else line
            x1, y1, x2, y2 = int(line_arr[0]), int(line_arr[1]), int(line_arr[2]), int(line_arr[3])
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            # 只保留接近水平或接近45度的线 (电力线常见角度)
            if angle < 30 or (60 < angle < 120):
                line_groups.append((min(x1,x2), min(y1,y2), 
                                   abs(x2-x1), abs(y2-y1)))
        
        # 合并重叠的线
        line_groups = sorted(line_groups, key=lambda b: (b[1], b[0]))
        merged = []
        for box in line_groups:
            if merged:
                last = merged[-1]
                # 如果y坐标接近，合并
                if abs(box[1] - last[1]) < 30 and abs(box[0] - last[0]) < 50:
                    new_x = min(last[0], box[0])
                    new_y = min(last[1], box[1])
                    new_w = max(last[0]+last[2], box[0]+box[2]) - new_x
                    new_h = max(last[1]+last[3], box[1]+box[3]) - new_y
                    merged[-1] = (new_x, new_y, new_w, new_h)
                    continue
            merged.append(box)
        
        for x, y, bw, bh in merged:
            if bw > 50 and bh < 30:  # 电力线是细长的
                boxes.append((1, x, y, bw, bh))  # class 1: power_line
    
    return boxes


def detect_ice(img):
    """检测覆冰 - 基于白色+纹理"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    # 白色区域 (低饱和度, 高亮度)
    low_sat = hsv[:, :, 1] < 60
    high_val = hsv[:, :, 2] > 160
    white_mask = (low_sat & high_val).astype(np.uint8) * 255
    
    # 纹理检测 (覆冰表面有纹理)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 80)
    edge_density = cv2.GaussianBlur((edges > 0).astype(np.float32), (15, 15), 0)
    has_texture = (edge_density > 0.03).astype(np.uint8) * 255
    
    # 冰区域 = 白色 + 有纹理
    ice_mask = cv2.bitwise_and(white_mask, has_texture)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel)
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > 20 and bh > 20:
            boxes.append((2, x, y, bw, bh))  # class 2: ice
    
    return boxes


def detect_tower(img):
    """检测铁塔 - 基于垂直边缘结构"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 检测垂直边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x).astype(np.uint8)
    
    # 阈值化
    _, binary = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
    
    # 形态学操作 - 强调垂直结构
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        # 铁塔是竖长的
        if aspect < 0.8 and bh > h * 0.15:
            boxes.append((3, x, y, bw, bh))  # class 3: tower
    
    return boxes


def box_to_yolo(cls_id, x, y, bw, bh, img_w, img_h):
    """转换为YOLO格式 (归一化的中心坐标+宽高)"""
    x_center = (x + bw / 2) / img_w
    y_center = (y + bh / 2) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def process_image(img_path, output_img_path, label_path):
    """处理单张图像，生成标注"""
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    all_boxes = []
    
    # 检测各类目标
    all_boxes.extend(detect_insulators(img))
    all_boxes.extend(detect_power_lines(img))
    all_boxes.extend(detect_ice(img))
    all_boxes.extend(detect_tower(img))
    
    # 过滤超出图像边界的框
    valid_boxes = []
    for cls_id, x, y, bw, bh in all_boxes:
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        if bw > 10 and bh > 10:
            valid_boxes.append((cls_id, x, y, bw, bh))
    
    # 保存图像
    cv2.imwrite(output_img_path, img)
    
    # 保存标注
    with open(label_path, 'w') as f:
        for cls_id, x, y, bw, bh in valid_boxes:
            yolo_line = box_to_yolo(cls_id, x, y, bw, bh, w, h)
            f.write(yolo_line + '\n')
    
    return len(valid_boxes)


def main():
    print("=" * 60)
    print("自动标注脚本 - 输电线路组件检测")
    print(f"类别: {CLASSES}")
    print("=" * 60)
    
    # 创建目录
    train_img_dir = os.path.join(OUTPUT_DIR, "train")
    val_img_dir = os.path.join(OUTPUT_DIR, "val")
    train_label_dir = os.path.join(LABEL_DIR, "train")
    val_label_dir = os.path.join(LABEL_DIR, "val")
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 收集所有图片
    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    images = sorted(images)
    
    if not images:
        print(f"错误: {INPUT_DIR} 中没有图片")
        return
    
    print(f"找到 {len(images)} 张图片")
    
    # 打乱并分割
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    
    # 统计
    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # 处理训练集
    print(f"\n标注训练集...")
    for i, img_path in enumerate(train_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # 清理文件名中的特殊字符
        safe_name = base_name.replace('.rf.', '_').replace('.', '_')
        
        ext = '.jpg'
        out_img = os.path.join(train_img_dir, f"{safe_name}{ext}")
        out_label = os.path.join(train_label_dir, f"{safe_name}.txt")
        
        n = process_image(img_path, out_img, out_label)
        total_boxes += n
        for cls_id in range(4):
            # 读取标注统计
            if os.path.exists(out_label):
                with open(out_label) as f:
                    for line in f:
                        cid = int(line.strip().split()[0])
                        class_counts[cid] = class_counts.get(cid, 0) + 1
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(train_images)}...")
    
    # 处理验证集
    print(f"\n标注验证集...")
    for i, img_path in enumerate(val_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        safe_name = base_name.replace('.rf.', '_').replace('.', '_')
        
        ext = '.jpg'
        out_img = os.path.join(val_img_dir, f"{safe_name}{ext}")
        out_label = os.path.join(val_label_dir, f"{safe_name}.txt")
        
        n = process_image(img_path, out_img, out_label)
        total_boxes += n
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(val_images)}...")
    
    # 生成 data.yaml
    yaml_content = f"""path: D:/dehaze_fusion/yolo_dataset
train: images/train
val: images/val

nc: 4
names: ['insulator', 'power_line', 'ice', 'tower']
"""
    yaml_path = os.path.join(r"D:\dehaze_fusion\yolo_dataset", "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print(f"标注完成!")
    print(f"  总标注框数: {total_boxes}")
    print(f"  各类别统计:")
    for cls_id, count in class_counts.items():
        print(f"    {CLASSES[cls_id]}: {count}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  训练集: {train_img_dir}")
    print(f"  验证集: {val_img_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
