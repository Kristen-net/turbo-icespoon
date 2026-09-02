"""路径参数化与配置加载测试 (P0-1/P0-2a)."""

from __future__ import annotations

import importlib

import pytest
import yaml

from icewave.train.config import deep_update, load_config
from icewave.utils.paths import checkpoint_path, dataset_root, expand


class TestPaths:
    def test_checkpoint_path_layout(self):
        p = checkpoint_path("m4")
        assert p.name == "m4_best.pth"
        assert "checkpoints" in str(p)

    def test_dataset_root(self):
        assert dataset_root("foo").name == "foo"

    def test_expand_env(self, monkeypatch):
        monkeypatch.setenv("ICEWAVE_TEST_VAR", "/tmp/abc")
        assert str(expand("$ICEWAVE_TEST_VAR/x")).replace("\\", "/").endswith("abc/x")

    def test_env_override_data_root(self, monkeypatch, tmp_path):
        """ICEWAVE_DATA_ROOT 环境变量应重定向 DATA_ROOT (云服务器部署的关键)."""
        import icewave.utils.paths as paths_mod
        monkeypatch.setenv("ICEWAVE_DATA_ROOT", str(tmp_path))
        mod2 = importlib.reload(paths_mod)
        try:
            assert mod2.DATA_ROOT == tmp_path
        finally:
            monkeypatch.delenv("ICEWAVE_DATA_ROOT")
            importlib.reload(paths_mod)


class TestConfig:
    def test_env_expansion_with_default(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(
            "data:\n  root: ${ICEWAVE_TEST_ROOT:/fallback/dir}\n"
            "plain: no_placeholder\n", encoding="utf-8")
        monkeypatch.delenv("ICEWAVE_TEST_ROOT", raising=False)
        cfg = load_config(cfg_path)
        assert cfg["data"]["root"] == "/fallback/dir"

    def test_env_expansion_with_value(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("root: ${ICEWAVE_TEST_ROOT:x}\n", encoding="utf-8")
        monkeypatch.setenv("ICEWAVE_TEST_ROOT", "/real/path")
        assert load_config(cfg_path)["root"] == "/real/path"

    def test_undefined_no_default_kept(self, tmp_path, monkeypatch):
        """无默认值的未定义变量保持原样 (可诊断而非静默吞错)."""
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("root: ${ICEWAVE_UNDEFINED_VAR_XYZ}\n", encoding="utf-8")
        monkeypatch.delenv("ICEWAVE_UNDEFINED_VAR_XYZ", raising=False)
        assert load_config(cfg_path)["root"] == "${ICEWAVE_UNDEFINED_VAR_XYZ}"

    def test_deep_update(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 4}
        out = deep_update(base, override)
        assert out == {"a": 1, "b": {"c": 99, "d": 3}, "e": 4}
        assert base["b"]["c"] == 2  # 不改原 dict

    def test_load_train_yaml_smoke(self, tmp_path):
        cfg_path = tmp_path / "t.yaml"
        yaml.safe_dump({"model": {"version": "m4"}, "train": {"epochs": 1}},
                       cfg_path.open("w", encoding="utf-8"))
        cfg = load_config(cfg_path)
        assert cfg["model"]["version"] == "m4"
