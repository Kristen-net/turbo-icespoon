"""IceWave 数据子包 (P0-4/P0-5/P1-1).

组件
----
- ``degradation``: 复合退化物理模型 (雾 + 覆冰合成, 含元数据)
- ``dataset``: IceAwareDataset (人工标注优先于伪标签, 走廊膨胀)
- ``build_dataset``: 数据集构建 CLI (场景级切分, 边界图排除, 浓雾档轮转)
"""

from icewave.data.degradation import (  # noqa: F401
    HAZE_PRESETS,
    HazeParams,
    IceParams,
    ice_composite,
    make_ice_thickness,
    make_ice_texture,
    make_transmission_map,
    synthesize_haze,
    synthesize_hazy_iced,
)
from icewave.data.dataset import IceAwareDataset, _pair_base  # noqa: F401