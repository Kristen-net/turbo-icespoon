import os

data_dir = r'D:\DATA_ALL'

if os.path.exists(data_dir):
    print(f'Path exists: {data_dir}')
    items = os.listdir(data_dir)
    print(f'Top-level items: {len(items)}')
    for item in items[:30]:
        full = os.path.join(data_dir, item)
        is_dir = os.path.isdir(full)
        prefix = 'DIR ' if is_dir else 'FILE'
        print(f'  [{prefix}] {item}')
    
    # Count images recursively
    total = 0
    dir_counts = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
                total += 1
                rel = os.path.relpath(root, data_dir)
                dir_counts[rel] = dir_counts.get(rel, 0) + 1
    print(f'\nTotal images: {total}')
    print(f'Directory structure:')
    for d, c in sorted(dir_counts.items()):
        print(f'  {d}: {c}')
else:
    print(f'Path does NOT exist: {data_dir}')
