"""检查每个zip里的图片是否可读取"""
import zipfile
import cv2
import numpy as np
import io
import os

data_dir = r'D:\DATA_ALL'

zips = [f for f in os.listdir(data_dir) if f.lower().endswith('.zip')]

for zname in zips:
    zpath = os.path.join(data_dir, zname)
    print(f'\n=== {zname} ===')
    with zipfile.ZipFile(zpath, 'r') as zf:
        names = [n for n in zf.namelist() 
                 if n.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
        print(f'  Total image entries: {len(names)}')
        
        readable = 0
        unreadable = 0
        sample_sizes = []
        
        for n in names[:20]:  # 只检查前20张
            try:
                data = zf.read(n)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    readable += 1
                    sample_sizes.append(img.shape[:2])
                else:
                    unreadable += 1
            except Exception as e:
                unreadable += 1
        
        print(f'  Checked first 20: {readable} readable, {unreadable} unreadable')
        if sample_sizes:
            print(f'  Sample sizes: {sample_sizes[:5]}')
        
        # 检查全部需要多久 - 估算
        # 如果前20张都能读, 假设大部分可读
