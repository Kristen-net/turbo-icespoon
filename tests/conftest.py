"""测试公共设施: src 路径注入 + 旧版模块按文件加载 (不入 sys.path)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
# 同时注入仓库根路径: 使 `from tests.conftest import ...` 在 pytest 以 `tests/` 为
# 收集参数时也能解析 (python -m pytest 会把 cwd 加入 sys.path, 直接 pytest 不会,
# 导致 CI 上 collect 阶段 ModuleNotFoundError: No module named 'tests' → exit code 2).
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module_from_file(name: str, rel_path: str):
    """从仓库内相对路径加载旧版模块 (importlib, 不污染 sys.path).

    旧代码普遍在模块级写死 sys.path.insert(D:\\...), 因此不能直接 import。
    """
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"旧模块不存在: {rel_path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def old_dehazeformer():
    """旧版 DehazeFormer (source_dehazeformer/dehazeformer.py)."""
    return load_module_from_file("old_dehazeformer", "source_dehazeformer/dehazeformer.py")


@pytest.fixture(scope="session")
def old_hawfe_v1():
    return load_module_from_file(
        "old_hawfe_v1", "phase4_dataset_baseline_20260815/ha_wfe.py")


@pytest.fixture(scope="session")
def old_hawfe_v2():
    return load_module_from_file(
        "old_hawfe_v2", "phase5_hawfe_training_20260816/ha_wfe_v2.py")


def old_integrate_hawfe_v2_with_prompt(model, old_hawfe_v2_mod,
                                       channels=96, prompt_channels=32):
    """逐字复刻 phase5_hawfe_training_20260816/clip_fog_prompt.py:168-210.

    无法直接 import 该文件 (其模块级代码写死 D:\\dehaze_fusion\\HazeCLIP 路径),
    故在测试中忠实复刻 monkey-patch 逻辑作为"旧行为"参照。
    """
    HAWFEv2 = old_hawfe_v2_mod.HAWFEv2

    hawfe = HAWFEv2(channels, prompt_channels=prompt_channels)
    device = next(model.parameters()).device
    hawfe = hawfe.to(device)
    model.hawfe = hawfe
    model.clip_prompt = None

    def new_forward_features(x):
        x = model.patch_embed(x)
        x = model.layer1(x)
        skip1 = x

        x = model.patch_merge1(x)
        x = model.layer2(x)
        skip2 = x

        x = model.patch_merge2(x)
        x = model.layer3(x)
        M_h = model.clip_prompt
        x = model.hawfe(x, M_h)
        x = model.patch_split1(x)

        x = model.fusion1([x, model.skip2(skip2)]) + x
        x = model.layer4(x)
        x = model.patch_split2(x)

        x = model.fusion2([x, model.skip1(skip1)]) + x
        x = model.layer5(x)
        x = model.patch_unembed(x)
        return x

    model.forward_features = new_forward_features
    return model
