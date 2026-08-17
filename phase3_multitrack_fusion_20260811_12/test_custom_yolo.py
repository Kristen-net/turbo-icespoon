"""用自定义YOLO模型检测融合去雾后的图片"""
import os, cv2, json
from ultralytics import YOLO

model = YOLO(r'D:\dehaze_fusion\yolo_train_output\power_line_yolo\weights\best.pt')
fusion_dir = r'D:\dehaze_fusion\fusion_3tracks_output'
out_dir = r'D:\dehaze_fusion\custom_yolo_output'
os.makedirs(out_dir, exist_ok=True)

results_list = []
for f in sorted(os.listdir(fusion_dir)):
    if not f.endswith('_fusion_final.png'):
        continue
    base = f.replace('_fusion_final.png', '')
    img_path = os.path.join(fusion_dir, f)
    results = model(img_path, conf=0.15, verbose=False)
    annotated = results[0].plot()
    n = len(results[0].boxes)
    classes = [results[0].names[int(c)] for c in results[0].boxes.cls.cpu().numpy()] if n > 0 else []
    cv2.imwrite(os.path.join(out_dir, base + '_custom_det.png'), annotated)
    print(base + ': ' + str(n) + ' targets = ' + str(classes))
    results_list.append({'name': base, 'count': n, 'classes': classes})

total_det = sum(r['count'] for r in results_list)
print('\nTotal images: ' + str(len(results_list)))
print('Total detections: ' + str(total_det))
