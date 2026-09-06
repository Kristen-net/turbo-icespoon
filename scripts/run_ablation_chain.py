"""链式消融实验: 等待当前训练进程结束后自动启动下一个实验。

用法:
    conda run -n dehaze_fusion python -c "
import sys, os; os.chdir(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon');
sys.path.insert(0, r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\src');
exec(open(r'C:\\Users\\2457025871\\.trae-cn\\work\\turbo-icespoon\\scripts\\run_ablation_chain.py').read())
    "
"""
import sys
import os
import time
import subprocess
import json
from pathlib import Path

REPO = Path(r"C:\Users\2457025871\.trae-cn\work\turbo-icespoon")
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

# 实验队列: (config_name, output_name)
EXPERIMENTS = [
    ("ablation_no_uncertainty", "ablation_no_uncertainty"),
    ("ablation_cascade", "ablation_cascade"),
]

LOG_FILE = REPO / "outputs" / "ablation_chain.log"

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def is_training_running():
    """检查是否有正在运行的训练进程 (python 进程占用大量内存)"""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | "
             "Where-Object { $_.WorkingSet64 -gt 100MB } | "
             "Select-Object -First 1 | Measure-Object | "
             "Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.strip()) > 0
    except:
        return False

def wait_for_training_done(check_interval=60):
    """等待当前训练进程结束"""
    log("等待当前训练进程结束...")
    idle_count = 0
    while idle_count < 3:
        if not is_training_running():
            idle_count += 1
            log(f"  未检测到训练进程 (连续 {idle_count}/3)")
            if idle_count < 3:
                time.sleep(check_interval)
        else:
            idle_count = 0
            time.sleep(check_interval)
    log("当前训练进程已结束")

def run_experiment(config_name, output_name):
    """运行单个消融实验"""
    log(f"=== 启动消融实验: {config_name} ===")

    # 检查配置文件是否存在
    cfg_path = REPO / "configs" / "train" / f"{config_name}.yaml"
    if not cfg_path.exists():
        log(f"  [错误] 配置文件不存在: {cfg_path}")
        return False

    # 构建 Python 内联执行命令
    run_script = REPO / "scripts" / "run_ablation.py"
    cmd = [
        "conda", "run", "--no-banner", "-n", "dehaze_fusion",
        "python", str(run_script),
        "--config", config_name,
        "--output", output_name,
    ]

    log(f"  命令: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        log(f"  退出码: {proc.returncode}")
        if proc.stdout:
            # 只记录最后 20 行
            lines = proc.stdout.strip().split("\n")
            for line in lines[-20:]:
                log(f"  [stdout] {line}")
        if proc.stderr:
            lines = proc.stderr.strip().split("\n")
            for line in lines[-10:]:
                log(f"  [stderr] {line}")

        # 检查输出
        metrics_path = REPO / "outputs" / output_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            log(f"  Best PSNR: {metrics.get('best_psnr', 'N/A')}")
            log(f"  Epochs: {len(metrics.get('history', []))}")
            return True
        else:
            log(f"  [警告] 未找到 metrics.json: {metrics_path}")
            return proc.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  [错误] 实验超时 (4 小时)")
        return False
    except Exception as e:
        log(f"  [错误] {e}")
        return False

# 主逻辑
log("=" * 60)
log("消融实验链式启动器")
log(f"待运行实验: {[e[0] for e in EXPERIMENTS]}")
log("=" * 60)

# 等待当前训练结束
wait_for_training_done(check_interval=60)

# 依次运行实验
for config_name, output_name in EXPERIMENTS:
    success = run_experiment(config_name, output_name)
    if success:
        log(f"实验 {config_name} 完成")
    else:
        log(f"实验 {config_name} 失败, 继续下一个")

    # 实验间冷却
    log("冷却 30 秒...")
    time.sleep(30)

log("=" * 60)
log("所有消融实验已完成")
log("=" * 60)

# 汇总结果
log("\n=== 消融实验汇总 ===")
all_outputs = ["joint_v2_w5", "ablation_no_boxfeat"]
for config_name, output_name in EXPERIMENTS:
    all_outputs.append(output_name)

for name in all_outputs:
    metrics_path = REPO / "outputs" / name / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            m = json.load(f)
        log(f"  {name:30s}  Best PSNR = {m.get('best_psnr', 0):.2f} dB  "
            f"Epochs = {len(m.get('history', []))}")
    else:
        log(f"  {name:30s}  [未找到]")
