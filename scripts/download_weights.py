#!/usr/bin/env python3
"""权重下载脚本 (含 SHA256 校验, P0-2c 重构后).

权重清单从 configs/weights.yaml 加载 (维护者可手工编辑或用
scripts/populate_manifest.py 自动生成)。

用法:
    python scripts/download_weights.py --all
    python scripts/download_weights.py --models m4 m2p
    python scripts/download_weights.py --hazeclip
    python scripts/download_weights.py --yolo

权重放置: 默认在项目根 weights/ 下, 可通过 ICEWAVE_WEIGHTS_DIR 覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "weights.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_dir() -> Path:
    env = os.environ.get("ICEWAVE_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "weights"


# ---- 极简 YAML 解析 (避免额外依赖; 文件可由 populate_manifest.py 生成) ----
def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    out: dict = {}
    cur_key: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*$", line)
        if m:
            cur_key = m.group(1)
            if cur_key == "manifest_version":
                cur_key = None
                continue
            out[cur_key] = {}
            continue
        if cur_key is None:
            continue
        sm = re.match(r"^\s+([A-Za-z_]+)\s*:\s*(.*?)\s*$", line)
        if sm:
            out[cur_key][sm.group(1)] = sm.group(2).strip('"').strip("'")
    return out


def download(name: str, entry: dict, weights_dir: Path) -> bool:
    url = entry.get("url", "")
    sha = entry.get("sha256", "")
    rel = entry.get("rel", f"checkpoints/{name}.pth")
    if not url:
        print(f"[跳过] {name}: MANIFEST 未配置 url (请维护者上传后更新 configs/weights.yaml)")
        return False
    dest = weights_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if sha and _sha256(dest) == sha:
            print(f"[已存在] {name}: {dest} (校验通过)")
            return True
        print(f"[已存在但校验失败] {name}: {dest}, 将重新下载")

    print(f"[下载] {name}: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        print(f"[错误] {name}: 下载失败 -> {e}")
        return False

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
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"清单路径 (默认 {DEFAULT_MANIFEST})")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[错误] 清单不存在: {args.manifest}", file=sys.stderr)
        return 2

    manifest = _load_yaml(args.manifest)

    targets = set(args.models)
    if args.all:
        targets = set(manifest.keys())
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
        if name not in manifest:
            print(f"[警告] 未知目标: {name}")
            continue
        ok = download(name, manifest[name], weights_dir) and ok
    if not ok:
        print("\n部分下载未通过校验, 请检查上方输出。")
        return 1
    print(f"\n权重目录: {weights_dir}")
    print("使用时设置: export ICEWAVE_WEIGHTS_DIR=<该目录>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
