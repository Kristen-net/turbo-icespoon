"""配置加载: YAML + ${ENV:default} 展开 + 默认值 (P0-1 路径参数化)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归展开字符串中的 ${VAR} / ${VAR:default} 占位符."""
    if isinstance(value, str):
        def repl(m):
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        prev = None
        while prev != value:  # 支持嵌套展开
            prev = value
            value = _ENV_PATTERN.sub(repl, value)
        return value
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """加载 YAML 配置并展开环境变量占位符."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 override 到 base (dict 深合并, 其余覆盖)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def git_commit() -> str:
    """当前 git commit (实验快照用; 非 git 环境返回 'unknown')."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[3]),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"
