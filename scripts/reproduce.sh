#!/usr/bin/env bash
# IceWave 一键端到端复现脚本 (SCI 投稿级 Reproducibility 配套)
#
# 用法:
#   bash scripts/reproduce.sh [STAGE]
#   STAGE = env | weights | dataset | train | eval | downstream | all (默认)
#
# 前置: 已设置 ICEWAVE_DATA_ROOT (含 train/val/test 子目录 + 标注),
#       已安装 conda 或 venv.
#
# 该脚本是"骨架脚本", 各 stage 给出命令而非真正执行 (避免无 GPU/数据时误跑)。
# 用户确认硬件就绪后, 把对应 stage 的 EXIT_DRY_RUN 注释掉即可真实执行。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-all}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ----- 可在执行前修改的环境变量 -----
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-icewave}"
DATA_ROOT="${ICEWAVE_DATA_ROOT:-$REPO_ROOT/data/raw}"
WEIGHTS_DIR="${ICEWAVE_WEIGHTS_DIR:-$REPO_ROOT/weights}"
OUTPUT_DIR="${ICEWAVE_OUTPUT_DIR:-$REPO_ROOT/outputs}"

# 设为 1 时, 该 stage 跳过执行 (默认: 0 真实执行; 改 STAGE=help 即可只看不跑)
DRY_RUN="${DRY_RUN:-0}"

run() {
    local desc="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        log "[DRY] $desc"
        log "       $*"
    else
        log "[RUN] $desc"
        eval "$@"
    fi
}

stage_env() {
    log "=== Stage 1/5: 创建 Python 环境 ==="
    run "conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y" \
        "conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y"
    run "安装 icewave 与全部 extras" \
        "conda activate $CONDA_ENV_NAME && pip install -e '.[all]'"
    run "验证安装" \
        "conda activate $CONDA_ENV_NAME && python -c 'import icewave, torch, ultralytics; print(\"icewave\", icewave.__version__ if hasattr(icewave, \"__version__\") else \"ok\")'"
}

stage_weights() {
    log "=== Stage 2/5: 下载预训练权重 ==="
    run "下载全部权重 (含 SHA256 校验)" \
        "python scripts/download_weights.py --all --manifest configs/weights.yaml"
    log "已下载权重位于 $WEIGHTS_DIR (或 configs/weights.yaml 中 rel 字段指定的子目录)"
}

stage_dataset() {
    log "=== Stage 3/5: 合成雾霾+覆冰退化数据集 ==="
    [[ -d "$DATA_ROOT" ]] || { log "[错误] 数据集不存在: $DATA_ROOT, 设置 ICEWAVE_DATA_ROOT"; return 1; }
    run "生成退化数据集" \
        "icewave-build-dataset --root $DATA_ROOT --config configs/benchmarks.yaml --output $OUTPUT_DIR/datasets"
}

stage_train() {
    log "=== Stage 4/5: 训练 m4 模型 (推荐; 耗时最长) ==="
    run "训练 m4 (HA-WFE + HazeCLIP 蒸馏 + ITL)" \
        "icewave-train --config configs/train/m4.yaml --data-root $DATA_ROOT --output-dir $OUTPUT_DIR/train/m4"
    log "其他模型: 把 m4 替换为 m1/m2/m2p/m3/joint"
}

stage_eval() {
    log "=== Stage 5a/5: 基准评测 (PSNR/SSIM/LPIPS) ==="
    run "在测试集上跑基准指标" \
        "icewave-eval-benchmark --weights $WEIGHTS_DIR/checkpoints/m4_best.pth --data-root $DATA_ROOT --output $OUTPUT_DIR/benchmarks"
}

stage_downstream() {
    log "=== Stage 5b/5: 下游检测 ΔmAP 评测 ==="
    run "在 VOC 2007 上评估去雾前后 mAP 增益" \
        "icewave-eval-downstream --weights $WEIGHTS_DIR/checkpoints/m4_best.pth --dataset voc --output $OUTPUT_DIR/downstream"
}

case "$STAGE" in
    env)       stage_env ;;
    weights)   stage_weights ;;
    dataset)   stage_dataset ;;
    train)     stage_train ;;
    eval)      stage_eval ;;
    downstream) stage_downstream ;;
    all)
        stage_env
        stage_weights
        stage_dataset
        stage_train
        stage_eval
        stage_downstream
        ;;
    help|-h|--help)
        cat <<EOF
用法: bash scripts/reproduce.sh [STAGE]
  STAGE = env | weights | dataset | train | eval | downstream | all
  DRY_RUN=1 仅打印命令不执行; 默认真实执行。

环境变量 (可覆盖):
  ICEWAVE_DATA_ROOT     数据集根 (默认: ./data/raw)
  ICEWAVE_WEIGHTS_DIR   权重目录 (默认: ./weights)
  ICEWAVE_OUTPUT_DIR    产物目录 (默认: ./outputs)
  PYTHON_VERSION        Python 版本 (默认: 3.11)
  CONDA_ENV_NAME        conda 环境名 (默认: icewave)
EOF
        ;;
    *)
        echo "未知 stage: $STAGE (运行 'bash scripts/reproduce.sh help' 查看帮助)"
        exit 2
        ;;
esac

log "✓ 完成 stage: $STAGE"
log "  下一步: 运行 'bash scripts/reproduce.sh help' 查看其他 stage,"
log "  或参考 docs/REPRODUCIBILITY.md 获取完整投稿级声明。"