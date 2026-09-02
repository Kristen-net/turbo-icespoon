import torch
import sys

path = r'c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth'
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print('Type:', type(ckpt))
print('Keys:', list(ckpt.keys()))
print('iteration:', ckpt.get('iteration'))
print('trainer:', type(ckpt.get('trainer')))

m = ckpt['model']
print('model type:', type(m))

if hasattr(m, 'state_dict'):
    sd = m.state_dict()
elif isinstance(m, dict):
    sd = m
else:
    sd = {}

print('sd type:', type(sd))
if isinstance(sd, dict):
    keys = list(sd.keys())
    print('Total params:', len(keys))
    print('--- first 40 ---')
    for k in keys[:40]:
        v = sd[k]
        print(f'  {k}: {v.shape if hasattr(v, "shape") else type(v)}')
    print('--- last 10 ---')
    for k in keys[-10:]:
        v = sd[k]
        print(f'  {k}: {v.shape if hasattr(v, "shape") else type(v)}')
    # try to find class info
    class_keys = [k for k in keys if 'class' in k.lower() or 'cls' in k.lower() or 'head' in k.lower() or 'num_classes' in k.lower()]
    if class_keys:
        print('--- class-related keys ---')
        for k in class_keys[:20]:
            v = sd[k]
            print(f'  {k}: {v.shape if hasattr(v, "shape") else type(v)}')
else:
    print('Not a dict, trying repr:')
    print(repr(m)[:2000])

# Also check if model object has useful attrs
if hasattr(m, 'names'):
    print('Model names:', m.names)
if hasattr(m, 'nc'):
    print('Model nc (num classes):', m.nc)
if hasattr(m, 'yaml'):
    print('Model yaml:', m.yaml)
