#!/usr/bin/env python3
"""权重下载脚本 (含 SHA256 校验, P0-2c).

权重文件不随仓库分发 (过大且涉及许可), 需通过本脚本或手动下载。
下载后自动校验 SHA256, 不匹配即报错并提示删除重试。

用法:
    python scripts/download_weights.py --all
    python scripts/download_weights.py --models m4 m2p
    python scripts/download_weights.py --hazeclip
    python scripts/download_weights.py --yolo

权重托管地址与 SHA256 由项目维护者填写 (见下方 MANIFEST)。
当前为占位符, 需替换为实际可用的 URL。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 权重清单: 维护者在此登记 (url, sha256, 目标相对路径)
# TODO(维护者): 替换为实际托管 URL 与 SHA256 (可放 GitHub Releases / HuggingFace)
# ---------------------------------------------------------------------------
MANIFEST = {
    # 去雾模型检查点 (对应 configs/train/*.yaml)
    "m1":   {"url": "", "sha256": "", "rel": "checkpoints/m1_best.pth"},
    "m2":   {"url": "", "sha256": "", "rel": "checkpoints/m2_best.pth"},
    "m2p":  {"url": "", "sha256": "", "rel": "checkpoints/m2p_best.pth"},
    "m3":   {"url": "", "sha256": "", "rel": "checkpoints/m3_best.pth"},
    "m4":   {"url": "", "sha256": "", "rel": "checkpoints/m4_best.pth"},
    "joint": {"url": "", "sha256": "", "rel": "checkpoints/joint_best.pth"},
    # HazeCLIP 教师权重 (蒸馏用, 可选)
    "hazeclip": {"url": "", "sha256": "", "rel": "hazeclip/model.pth"},
    # YOLO 覆冰检测权重 (AGPL-3.0, 见 NOTICE.md)
    "yolo":     {"url": "", "sha256": "", "rel": "yolo/power_line_best.pt"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_dir() -> Path:
    import os
    env = os.environ.get("ICEWAVE_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "weights"


def download(name: str, entry: dict, weights_dir: Path) -> bool:
    url, sha = entry["url"], entry["sha256"]
    if not url:
        print(f"[跳过] {name}: 未配置下载 URL (维护者需在 MANIFEST 填写)")
        return False
    dest = weights_dir / entry["rel"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if sha and _sha256(dest) == sha:
            print(f"[已存在] {name}: {dest} (校验通过)")
            return True
        print(f"[已存在但校验失败] {name}: {dest}, 将重新下载")

    print(f"[下载] {name}: {url}")
    urllib.request.urlretrieve(url, str(dest))

    if sha:
        got = _sha256(dest)
        if got != sha:
            print(f"[错误] {name}: SHA256 不匹配!\n  期望 {sha}\n  实际 {got}")
            print("  请删除该文件后重试, 或联系维护者更新 MANIFEST。")
            return False
        print(f"[校验通过] {name}: {dest}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="IceWave 权重下载 (SHA256 校验)")
    ap.add_argument("--all", action="store_true", help="下载全部")
    ap.add_argument("--models", nargs="*", default=[],
                    help="下载指定去雾模型 (如 m4 m2p)")
    ap.add_argument("--hazeclip", action="store_true")
    ap.add_argument("--yolo", action="store_true")
    args = ap.parse_args(argv)

    targets = set(args.models)
    if args.all:
        targets = set(MANIFEST.keys())
    if args.hazeclip:
        targets.add("hazeclip")
    if args.yolo:
        targets.add("yolo")
    if not targets:
        ap.error("请指定 --all / --models / --hazeclip / --yolo")

    weights_dir = _weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in sorted(targets):
        if name not in MANIFEST:
            print(f"[警告] 未知目标: {name}")
            continue
        ok = download(name, MANIFEST[name], weights_dir) and ok
    if not ok:
        print("\n部分下载未通过校验, 请检查上方输出。")
        return 1
    print(f"\n权重目录: {weights_dir}")
    print("使用时设置: export ICEWAVE_WEIGHTS_DIR=<该目录>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
