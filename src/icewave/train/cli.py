"""训练入口: icewave-train --config configs/train/m4.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

from icewave.train.config import load_config
from icewave.train.trainer import Trainer


def main(argv=None):
    ap = argparse.ArgumentParser(description="IceWave 训练")
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--override", nargs="*", default=[],
                    help="覆盖项, 格式 key.subkey=value (JSON 值)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for kv in args.override:
        key, value = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            import json
            node[parts[-1]] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            node[parts[-1]] = value

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
