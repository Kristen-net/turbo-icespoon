"""Read TensorBoard logs for M2' training"""
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(r"D:\dehaze_fusion\icewave_output\m2p_hawfe_v2\logs")
ea.Reload()

tags = ea.Tags()
print("Available tags:", tags)

if 'scalars' in tags:
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        print(f"\n{tag}:")
        for e in events:
            print(f"  step={e.step} value={e.value:.6f}")
