"""模型层重构兼容性测试 (P0-1a 的验收标准).

验证三件事 (对应 hawfe.py docstring 的承诺):
1. 参数键名新旧一致 → 旧 M2/M2p/M3/M4 检查点可直接 ``load_state_dict(strict=True)``;
2. 数值行为新旧一致 → 同权重同输入, 输出逐元素相等;
3. 新增能力 (显式 fog_prompt 传参 / 非整数倍尺寸) 不改变旧行为。
"""

from __future__ import annotations

import pytest
import torch

from icewave.models import HAWFE, HAWFEv2, IceWaveDehazeFormer, build_model
from tests.conftest import old_integrate_hawfe_v2_with_prompt

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. HA-WFE 模块: 键名与数值等价
# ---------------------------------------------------------------------------
class TestHAWFECompat:
    def test_v2_param_keys_identical(self, old_hawfe_v2):
        old = old_hawfe_v2.HAWFEv2(96, prompt_channels=32)
        new = HAWFEv2(96, prompt_channels=32)
        old_keys, new_keys = set(old.state_dict()), set(new.state_dict())
        assert old_keys == new_keys, (
            f"键名不一致: 仅旧={old_keys - new_keys}, 仅新={new_keys - old_keys}")

    def test_v1_param_keys_identical(self, old_hawfe_v1):
        old = old_hawfe_v1.HAWFE(96)
        new = HAWFE(96)
        assert set(old.state_dict()) == set(new.state_dict())

    def test_v2_init_values_identical(self, old_hawfe_v2):
        """可学习标量初值 (alpha/beta) 必须与旧实现一致 (影响训练起点)."""
        old, new = old_hawfe_v2.HAWFEv2(96), HAWFEv2(96)
        for k in ("alpha_ll", "alpha_lh", "alpha_hl", "alpha_hh", "beta"):
            ov = old.state_dict()[k].item()
            nv = new.state_dict()[k].item()
            assert ov == nv, f"{k}: 旧={ov} 新={nv}"

    def test_v2_forward_numerical_match(self, old_hawfe_v2):
        old, new = old_hawfe_v2.HAWFEv2(96, prompt_channels=32), \
            HAWFEv2(96, prompt_channels=32)
        new.load_state_dict(old.state_dict(), strict=True)

        x = torch.rand(2, 96, 32, 32)
        with torch.no_grad():
            y_old = old(x)
            y_new = new(x)
        torch.testing.assert_close(y_old, y_new, rtol=0, atol=0,
                                   msg="HAWFEv2 新旧前向应逐位一致")

    def test_v2_forward_match_with_prompt(self, old_hawfe_v2):
        old, new = old_hawfe_v2.HAWFEv2(96, prompt_channels=32), \
            HAWFEv2(96, prompt_channels=32)
        new.load_state_dict(old.state_dict(), strict=True)

        x = torch.rand(2, 96, 32, 32)
        m = torch.rand(2, 32, 7, 7)
        with torch.no_grad():
            torch.testing.assert_close(old(x, m), new(x, m), rtol=0, atol=0)

    def test_odd_size_padding(self):
        """奇数空间尺寸走 reflect padding 路径, 不应报错."""
        m = HAWFEv2(96)
        x = torch.rand(1, 96, 33, 31)
        assert m(x).shape == x.shape

    def test_backward_flows(self):
        m = HAWFEv2(96)
        x = torch.rand(1, 96, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# 2. 完整模型: 旧 monkey-patch vs 新子类
# ---------------------------------------------------------------------------
class TestFullModelCompat:
    @pytest.fixture()
    def old_new_pair(self, old_dehazeformer, old_hawfe_v2):
        old_model = old_dehazeformer.dehazeformer_s()
        old_model = old_integrate_hawfe_v2_with_prompt(
            old_model, old_hawfe_v2, channels=96, prompt_channels=32)
        new_model = build_model("m4")
        return old_model, new_model

    def test_state_dict_keys_identical(self, old_new_pair):
        old_model, new_model = old_new_pair
        old_keys, new_keys = set(old_model.state_dict()), set(new_model.state_dict())
        assert old_keys == new_keys, (
            f"检查点兼容性破坏: 仅旧={sorted(old_keys - new_keys)[:5]}, "
            f"仅新={sorted(new_keys - old_keys)[:5]}")

    def test_old_checkpoint_strict_load(self, old_new_pair):
        """核心验收: 旧 monkey-patch 模型导出的 state_dict 能 strict 加载进新类."""
        old_model, new_model = old_new_pair
        new_model.load_state_dict(old_model.state_dict(), strict=True)

    def test_forward_identical_no_prompt(self, old_new_pair):
        old_model, new_model = old_new_pair
        new_model.load_state_dict(old_model.state_dict(), strict=True)
        old_model.eval(), new_model.eval()

        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            y_old = old_model(x)          # clip_prompt=None
            y_new = new_model(x)          # fog_prompt=None
        torch.testing.assert_close(y_old, y_new, rtol=1e-6, atol=1e-6)

    def test_forward_identical_with_prompt(self, old_new_pair):
        old_model, new_model = old_new_pair
        new_model.load_state_dict(old_model.state_dict(), strict=True)
        old_model.eval(), new_model.eval()

        x = torch.rand(1, 3, 256, 256)
        m = torch.rand(1, 32, 7, 7)
        with torch.no_grad():
            old_model.clip_prompt = m     # 旧: 全局副作用通道
            y_old = old_model(x)
            y_new = new_model(x, fog_prompt=m)  # 新: 显式传参
        torch.testing.assert_close(y_old, y_new, rtol=1e-6, atol=1e-6)

    def test_legacy_clip_prompt_attribute_still_works(self, old_new_pair):
        """旧调用习惯 model.clip_prompt = m 在新类中同样有效 (兼容承诺)."""
        _, new_model = old_new_pair
        x = torch.rand(1, 3, 128, 128)
        m = torch.rand(1, 32, 7, 7)
        with torch.no_grad():
            new_model.clip_prompt = m
            y_attr = new_model(x)
            y_arg = new_model(x, fog_prompt=m)
        torch.testing.assert_close(y_attr, y_arg, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. 工厂
# ---------------------------------------------------------------------------
class TestBuildModel:
    @pytest.mark.parametrize("version,has_hawfe", [
        ("m1", False), ("m2", True), ("m2p", True), ("m3", True), ("m4", True),
    ])
    def test_versions_build(self, version, has_hawfe):
        m = build_model(version)
        assert hasattr(m, "hawfe") == has_hawfe

    def test_m2_uses_hawfe_v1(self):
        from icewave.models.hawfe import HAWFE as V1
        assert isinstance(build_model("m2").hawfe, V1)

    def test_m4_uses_hawfe_v2(self):
        from icewave.models.hawfe import HAWFEv2 as V2
        assert isinstance(build_model("m4").hawfe, V2)

    def test_m2_hawfe_keys_match_old_v1(self, old_hawfe_v1):
        new_m2 = build_model("m2")
        old_v1 = old_hawfe_v1.HAWFE(96)
        new_keys = {k.removeprefix("hawfe.") for k in new_m2.state_dict()
                    if k.startswith("hawfe.")}
        assert new_keys == set(old_v1.state_dict())

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="未知模型版本"):
            build_model("m99")

    def test_all_versions_forward(self):
        x = torch.rand(1, 3, 64, 64)
        for v in ("m1", "m2", "m2p", "m3", "m4", "joint"):
            m = build_model(v)
            with torch.no_grad():
                assert m(x).shape == x.shape
