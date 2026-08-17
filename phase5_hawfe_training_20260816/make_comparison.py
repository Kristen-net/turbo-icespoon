import sys
import os
import cv2
import numpy as np

INPUT_DIR = r"D:\dehaze_fusion\my_test\input"
OUTPUT_DIR = r"D:\dehaze_fusion\my_test\output"
COMPARE_DIR = r"D:\dehaze_fusion\my_test\compare"
os.makedirs(COMPARE_DIR, exist_ok=True)

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)])

for fname in input_files:
    inp = os.path.join(INPUT_DIR, fname)
    out = os.path.join(OUTPUT_DIR, fname)
    mask_name = os.path.splitext(fname)[0] + "_ice_mask" + os.path.splitext(fname)[1]
    mask_path = os.path.join(OUTPUT_DIR, mask_name)

    if not os.path.exists(out):
        continue

    orig = cv2.imread(inp)
    dehazed = cv2.imread(out)

    if orig is None or dehazed is None:
        continue

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

    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_r = resize_pad(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), h, w)

        mask_overlay = dehazed_r.copy()
        mask_bool = cv2.resize(mask, (dehazed_r.shape[1], dehazed_r.shape[0])) > 127
        red_layer = np.zeros_like(dehazed_r)
        red_layer[:] = [0, 0, 255]
        mask_overlay = np.where(mask_bool[:, :, None], cv2.addWeighted(dehazed_r, 0.6, red_layer, 0.4, 0), dehazed_r)

        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r, gap, mask_overlay])
        labels = ["Original", "Dehazed", "Ice Detection (Red Overlay)"]
    else:
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([orig_r, gap, dehazed_r])
        labels = ["Original", "Dehazed"]

    label_h = 30
    label_bar = np.full((label_h, combined.shape[1], 3), 240, dtype=np.uint8)
    if len(labels) == 3:
        cw = w
        for i, label in enumerate(labels):
            cx = i * (w + 10) + w // 2
            cv2.putText(label_bar, label, (cx - len(label) * 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    else:
        for i, label in enumerate(labels):
            cx = i * (w + 10) + w // 2
            cv2.putText(label_bar, label, (cx - len(label) * 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    final = np.vstack([label_bar, combined])
    out_path = os.path.join(COMPARE_DIR, f"compare_{fname}")
    cv2.imwrite(out_path, final)
    print(f"[OK] {fname}")

print(f"\nAll comparisons saved to: {COMPARE_DIR}")
