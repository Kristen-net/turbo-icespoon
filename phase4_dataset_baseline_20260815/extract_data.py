import zipfile
import os

data_dir = r'D:\DATA_ALL'
extract_dir = r'D:\DATA_ALL\extracted'
os.makedirs(extract_dir, exist_ok=True)

zips = [f for f in os.listdir(data_dir) if f.lower().endswith('.zip')]
print(f'Found {len(zips)} zip files:')
for z in zips:
    print(f'  {z}')

total_files = 0
for zname in zips:
    zpath = os.path.join(data_dir, zname)
    zbasename = os.path.splitext(zname)[0]
    out_dir = os.path.join(extract_dir, zbasename)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f'\nExtracting {zname} ...')
    with zipfile.ZipFile(zpath, 'r') as zf:
        names = zf.namelist()
        img_count = sum(1 for n in names if n.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')))
        print(f'  Total entries: {len(names)}, images: {img_count}')
        total_files += img_count
        zf.extractall(out_dir)

print(f'\nTotal images across all zips: {total_files}')
print(f'Extracted to: {extract_dir}')
