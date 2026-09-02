import torch
import json

path = r'c:\Users\2457025871\.trae-cn\attachments\6a7018ff8440e8370e6f184d\10980b66-b2d0-4240-bb98-31bf3acabc9b_d9be178e-2805-43d9-a290-ac752e62bbca_model_0004999.pth'
ckpt = torch.load(path, map_location='cpu', weights_only=False)

trainer = ckpt.get('trainer', {})
print('=== Trainer dict ===')
print('Type:', type(trainer))
if isinstance(trainer, dict):
    print('Keys:', list(trainer.keys()))
    for k, v in trainer.items():
        vstr = str(v)
        if len(vstr) > 500:
            vstr = vstr[:500] + '...'
        print(f'  {k}: {vstr}')
else:
    print('Value:', str(trainer)[:2000])

# Check if model has config metadata
m = ckpt['model']
if isinstance(m, dict):
    meta_keys = [k for k in m.keys() if 'meta' in k.lower() or 'config' in k.lower()]
    print('\nMeta keys:', meta_keys)

# Check the backbone more carefully to determine ResNet version
sd = m if isinstance(m, dict) else {}
# Count residual blocks to determine ResNet depth
res_layers = set()
for k in sd.keys():
    if 'bottom_up.res' in k:
        layer = k.split('bottom_up.')[1].split('.')[0]
        res_layers.add(layer)
print('\nResNet layers:', sorted(res_layers))

# Count blocks in each layer
for layer in sorted(res_layers):
    blocks = set()
    for k in sd.keys():
        if f'bottom_up.{layer}.' in k:
            block_id = k.split(f'bottom_up.{layer}.')[1].split('.')[0]
            blocks.add(block_id)
    print(f'  {layer}: {len(blocks)} blocks -> {sorted(blocks)}')

# Determine ResNet version from block counts
# ResNet-50: [3, 4, 6, 3]
# ResNet-101: [3, 4, 23, 3]
# ResNet-152: [3, 8, 36, 3]
block_counts = []
for layer in sorted(res_layers):
    blocks = set()
    for k in sd.keys():
        if f'bottom_up.{layer}.' in k:
            block_id = k.split(f'bottom_up.{layer}.')[1].split('.')[0]
            blocks.add(block_id)
    block_counts.append(len(blocks))

print(f'\nBlock counts: {block_counts}')
if block_counts == [3, 4, 6, 3]:
    print('Backbone: ResNet-50')
elif block_counts == [3, 4, 23, 3]:
    print('Backbone: ResNet-101')
elif block_counts == [3, 8, 36, 3]:
    print('Backbone: ResNet-152')

print(f'\nNum classes (cls_score): {sd["roi_heads.box_predictor.cls_score.weight"].shape[0]}')
print(f'Mask channels: {sd["roi_heads.mask_head.predictor.weight"].shape[0]}')
print(f'RPN anchors: {sd["proposal_generator.rpn_head.objectness_logits.weight"].shape[0]}')
