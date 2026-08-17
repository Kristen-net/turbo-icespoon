import cv2, numpy as np

dir_path = r'D:\dehaze_fusion\fusion_3tracks_output'
files = ['ice1189_L1_pixel.png', 'ice1189_L2_feature.png', 'ice1189_L3_decision.png', 'ice1189_L4_cascade.png', 'ice1189_fusion_final.png']

imgs = {}
for f in files:
    img = cv2.imread(dir_path + '\\' + f)
    imgs[f] = img
    print(f'{f}: shape={img.shape}, mean={img.mean():.2f}, std={img.std():.2f}')

print()
print('=== 像素差异 (MAE = 平均绝对误差) ===')
keys = list(imgs.keys())
short = {'ice1189_L1_pixel.png': 'L1', 'ice1189_L2_feature.png': 'L2', 'ice1189_L3_decision.png': 'L3', 'ice1189_L4_cascade.png': 'L4', 'ice1189_fusion_final.png': 'final'}
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        diff = np.abs(imgs[keys[i]].astype(np.float64) - imgs[keys[j]].astype(np.float64))
        mae = diff.mean()
        max_diff = diff.max()
        same_pct = (diff < 1).mean() * 100
        print(f'{short[keys[i]]} vs {short[keys[j]]}: MAE={mae:.4f}, MaxDiff={max_diff:.1f}, 相同像素={same_pct:.1f}%')
