"""路径解析: 消除硬编码绝对路径 (P0-1).

优先级: 显式参数 > 环境变量 > 仓库内默认目录。
所有模块一律通过本模块取路径, 禁止写死盘符。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录 = src/icewave/utils/paths.py 向上三级
REPO_ROOT = Path(__file__).resolve().parents[3]

_ENV_KEYS = {
    "data": "ICEWAVE_DATA_ROOT",
    "weights": "ICEWAVE_WEIGHTS_DIR",
    "output": "ICEWAVE_OUTPUT_DIR",
    "clip": "ICEWAVE_CLIP_DIR",
    "hazeclip_weights": "ICEWAVE_HAZECLIP_WEIGHTS",
}


def _resolve(kind: str, default: Path) -> Path:
    env = os.environ.get(_ENV_KEYS[kind])
    if env:
        return Path(env).expanduser().resolve()
    return default


DATA_ROOT = _resolve("data", REPO_ROOT / "data")
WEIGHTS_DIR = _resolve("weights", REPO_ROOT / "weights")
OUTPUT_DIR = _resolve("output", REPO_ROOT / "outputs")
CLIP_DIR = _resolve("clip", REPO_ROOT / "source_hazeclip")
HAZECLIP_WEIGHTS = _resolve(
    "hazeclip_weights", WEIGHTS_DIR / "hazeclip" / "model.pth"
)


def dataset_root(name: str = "dataset") -> Path:
    """标准数据集根目录: $DATA_ROOT/<name>."""
    return DATA_ROOT / name


def checkpoint_path(version: str) -> Path:
    """各版本模型检查点默认位置: $WEIGHTS_DIR/checkpoints/<version>_best.pth."""
    return WEIGHTS_DIR / "checkpoints" / f"{version}_best.pth"


def expand(path: str | Path) -> Path:
    """展开用户目录与环境变量并取绝对路径."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
