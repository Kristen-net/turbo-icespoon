#!/usr/bin/env python3
"""权重清单生成器 (P0-2c 重构辅助工具).

读取本地训练输出目录, 计算每个 .pth / .pt 的 SHA256 + 大小, 写入或更新
configs/weights.yaml 中对应的条目。

典型用法:
    # 训练得到 m4/best.pth, m4/last.pth 后, 准备发布权重
    python scripts/populate_manifest.py \\
        --root outputs/train \\
        --base-url https://huggingface.co/Kristen-net/icewave-weights/resolve/main

参数:
    --root        包含 m1/ m2/ m2p/ m3/ m4/ joint/ 子目录的训练输出根
    --base-url    上传后的公网直链前缀 (HF Hub resolve/main 或 Releases 下载点)
    --pick        默认 "best", 可改为 "last"
    --yaml        清单路径 (默认 configs/weights.yaml)
    --dry-run     仅打印不写

本脚本不会自动上传; 维护者需自行把文件传到 HF / Releases 后, 再 git commit YAML。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "weights.yaml"

KNOWN_MODELS = ["m1", "m2", "m2p", "m3", "m4", "joint"]

# 每个模型期望的文件名 (按 --pick 切换)
EXPECTED_FILENAMES = {
    "best": "best.pth",
    "last": "last.pth",
    "epoch_best": "best.pth",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_yaml_simple(text: str) -> dict:
    """极简 YAML 解析: 仅支持本工具生成的 manifest 顶层 key / url+hash+rel 结构.
    避免引入 pyyaml 依赖; 文件本身就由本工具写入.
    """
    import re
    out: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*$", ln)
        if not m:
            i += 1
            continue
        key = m.group(1)
        if key in ("manifest_version",):
            i += 1
            continue
        # 收集子条目直到下一个顶层 key
        sub: dict = {}
        i += 1
        while i < len(lines):
            ln2 = lines[i]
            if re.match(r"^[A-Za-z_][\w-]*\s*:\s*$", ln2):
                break
            sm = re.match(r"^\s+([A-Za-z_]+)\s*:\s*(.*?)\s*$", ln2)
            if sm:
                sub[sm.group(1)] = sm.group(2).strip('"').strip("'")
            i += 1
        out[key] = sub
    return out


def _render_yaml(data: dict, header: str) -> str:
    lines = [header.rstrip("\n"), "", "manifest_version: 1", ""]
    # 顺序保留: m1...joint 再 hazeclip / yolo
    order = [k for k in KNOWN_MODELS if k in data] + [k for k in data if k not in KNOWN_MODELS]
    for k in order:
        sub = data[k]
        lines.append(f"{k}:")
        for fk in ("url", "sha256", "rel", "size_bytes", "note"):
            if fk in sub and sub[fk] != "":
                val = str(sub[fk])
                # 简单引号避免特殊字符
                if any(ch in val for ch in (":", "#", "'", '"')) or val.startswith("http"):
                    val = '"' + val.replace('"', '\\"') + '"'
                lines.append(f"  {fk}: {val}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="根据本地权重生成 MANIFEST YAML")
    ap.add_argument("--root", type=Path, required=True,
                    help="训练输出根 (含 m1/, m2/, ... 子目录)")
    ap.add_argument("--base-url", required=True,
                    help="公网前缀 (https://huggingface.co/.../resolve/main)")
    ap.add_argument("--pick", default="best", choices=list(EXPECTED_FILENAMES),
                    help="选 best / last (默认 best)")
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML,
                    help="清单路径")
    ap.add_argument("--dry-run", action="store_true", help="仅打印统计")
    args = ap.parse_args(argv)

    if not args.yaml.exists():
        print(f"[错误] 找不到清单: {args.yaml}", file=sys.stderr)
        return 1

    text = args.yaml.read_text(encoding="utf-8")
    header_lines = []
    for line in text.splitlines():
        if line.startswith("manifest_version"):
            break
        header_lines.append(line)

    data = _parse_yaml_simple(text)
    picked = EXPECTED_FILENAMES[args.pick]

    print(f"[扫描] root={args.root}")
    updates = 0
    for name in KNOWN_MODELS:
        sub_dir = args.root / name
        cand = sub_dir / picked
        if not cand.exists():
            print(f"  [跳过] {name}: 未找到 {cand}")
            continue
        sha = _sha256(cand)
        size = cand.stat().st_size
        url = args.base_url.rstrip("/") + "/" + name + "/" + cand.name
        data[name] = {
            "url": url,
            "sha256": sha,
            "rel": f"checkpoints/{name}_{args.pick}.pth",
            "size_bytes": size,
            "note": data.get(name, {}).get("note", f"configs/train/{name}.yaml"),
        }
        print(f"  [✓] {name}: {url} ({size/1e6:.1f}MB, sha256={sha[:12]}...)")
        updates += 1

    if updates == 0:
        print("\n未发现任何权重; 请检查 --root 是否包含 m1/, m2/, ... 子目录。",
              file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\n[DRY-RUN] 将更新 {updates} 条; --yaml {args.yaml} 未修改。")
        return 0

    new_text = _render_yaml(data, "\n".join(header_lines))
    args.yaml.write_text(new_text, encoding="utf-8")
    print(f"\n[OK] 已更新 {args.yaml} ({updates} 条)。")
    print("下一步: 把对应文件上传到 base-url 对应的存储后, git commit 此 YAML。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
