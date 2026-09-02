"""IceWave-DehazeFormer: 雾天输电线路去雾与覆冰检测研究框架.

包化重构后的标准入口。历史 phase*/ 目录保留为开发存档, 新功能一律在本包内实现。
"""

__version__ = "0.2.0"

from icewave.models import build_model, load_checkpoint  # noqa: F401
