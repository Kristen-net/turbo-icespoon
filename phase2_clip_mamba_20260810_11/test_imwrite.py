import cv2, numpy as np, os
test_dir = r'D:\dehaze_fusion\WDMamba\output\wdmamba_results'
os.makedirs(test_dir, exist_ok=True)
test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
test_path = os.path.join(test_dir, 'test_write.png')
ret = cv2.imwrite(test_path, test_img)
print(f'cv2.imwrite 返回: {ret}')
print(f'文件存在: {os.path.exists(test_path)}')
print(f'目录内容: {os.listdir(test_dir)}')
