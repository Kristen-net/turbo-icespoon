"""
自动化流水线: 监控M1 → 启动M2 → 生成对比报告
检测条件: M1日志出现"训练完成" → 启动M2
M2完成后自动生成对比报告
"""

import subprocess
import sys
import os
import time
import re

PYTHON = r"C:\Users\2457025871\miniconda3\envs\yolov8\python.exe"
M1_LOG = r"C:\Users\245702~1\AppData\Local\Temp\trae-agent-toolhost\jobs\job-9cef1267b74f4a07bac7667d6fee5b8d\output.log"
M2_SCRIPT = r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d\train_m2.py"
M2_LOG = r"D:\dehaze_fusion\icewave_output\m2_hawfe\training.log"
REPORT_FILE = r"D:\dehaze_fusion\icewave_output\ablation_report.txt"
STATUS_FILE = r"D:\dehaze_fusion\icewave_output\pipeline_status.txt"

def write_status(status):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write(status + "\n")
        f.write(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(status, flush=True)

def parse_epoch_lines(log_path):
    """从日志中提取所有epoch的PSNR/SSIM"""
    results = []
    if not os.path.exists(log_path):
        return results
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r'Epoch (\d+)/\d+ avg_loss=([\d.]+) val_PSNR=([\d.]+) val_SSIM=([\d.]+)', line)
            if m:
                results.append({
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "psnr": float(m.group(3)),
                    "ssim": float(m.group(4)),
                })
    return results

def wait_for_m1():
    """等待M1训练完成"""
    write_status("等待M1训练完成...")
    
    while True:
        if os.path.exists(M1_LOG):
            with open(M1_LOG, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "训练完成" in content:
                    write_status("M1训练已完成! 准备启动M2...")
                    return True
                # 检查M1是否还在运行
                m1_results = parse_epoch_lines(M1_LOG)
                if m1_results:
                    latest = m1_results[-1]
                    write_status(f"M1训练中: Epoch {latest['epoch']}/100, "
                               f"当前PSNR={latest['psnr']:.2f}, SSIM={latest['ssim']:.4f}")
        time.sleep(60)  # 每分钟检查一次

def run_m2():
    """启动M2训练"""
    os.makedirs(os.path.dirname(M2_LOG), exist_ok=True)
    write_status("M2训练启动中 (DehazeFormer-S + HA-WFE)...")
    
    # 启动M2训练
    cmd = [PYTHON, "-u", M2_SCRIPT]
    with open(M2_LOG, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
    
    # 监控M2进度
    while proc.poll() is None:
        m2_results = parse_epoch_lines(M2_LOG)
        if m2_results:
            latest = m2_results[-1]
            write_status(f"M2训练中: Epoch {latest['epoch']}/100, "
                       f"当前PSNR={latest['psnr']:.2f}, SSIM={latest['ssim']:.4f}")
        time.sleep(60)
    
    write_status("M2训练已完成! 生成对比报告中...")
    return True

def generate_report():
    """生成M1 vs M2消融对比报告"""
    m1_results = parse_epoch_lines(M1_LOG)
    m2_results = parse_epoch_lines(M2_LOG)
    
    if not m1_results or not m2_results:
        write_status("错误: 无法获取训练结果")
        return
    
    # 找最佳PSNR
    m1_best = max(m1_results, key=lambda x: x["psnr"])
    m2_best = max(m2_results, key=lambda x: x["psnr"])
    
    # 最后10个epoch的平均 (稳定性)
    m1_last10 = m1_results[-10:] if len(m1_results) >= 10 else m1_results
    m2_last10 = m2_results[-10:] if len(m2_results) >= 10 else m2_results
    
    m1_avg_psnr = sum(r["psnr"] for r in m1_last10) / len(m1_last10)
    m2_avg_psnr = sum(r["psnr"] for r in m2_last10) / len(m2_last10)
    m1_avg_ssim = sum(r["ssim"] for r in m1_last10) / len(m1_last10)
    m2_avg_ssim = sum(r["ssim"] for r in m2_last10) / len(m2_last10)
    
    # HA-WFE参数变化 (从M2日志提取)
    hawfe_params = []
    if os.path.exists(M2_LOG):
        with open(M2_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r'\[a_ll=([\d.-]+) a_hf=([\d.-]+) b=([\d.-]+)\]', line)
                if m:
                    hawfe_params.append({
                        "a_ll": float(m.group(1)),
                        "a_hf": float(m.group(2)),
                        "beta": float(m.group(3)),
                    })
    
    report = []
    report.append("=" * 60)
    report.append("消融实验对比报告: M1 (基线) vs M2 (+HA-WFE)")
    report.append("=" * 60)
    report.append("")
    report.append("模型配置:")
    report.append(f"  M1: DehazeFormer-S (1.28M参数)")
    report.append(f"  M2: DehazeFormer-S + HA-WFE (1.31M参数, +2.3%)")
    report.append(f"  训练: {config_epochs} epochs, batch=8, patch=192, lr=2e-4")
    report.append(f"  数据: 1266训练对, 84验证对")
    report.append("")
    report.append("最佳指标:")
    report.append(f"  {'指标':<15} {'M1基线':>10} {'M2+HA-WFE':>10} {'差异':>10}")
    report.append(f"  {'─'*45}")
    report.append(f"  {'最佳PSNR':<15} {m1_best['psnr']:>10.2f} {m2_best['psnr']:>10.2f} {m2_best['psnr']-m1_best['psnr']:>+10.2f}")
    report.append(f"  {'最佳SSIM':<15} {m1_best['ssim']:>10.4f} {m2_best['ssim']:>10.4f} {m2_best['ssim']-m1_best['ssim']:>+10.4f}")
    report.append(f"  {'最佳epoch':<15} {m1_best['epoch']:>10} {m2_best['epoch']:>10}")
    report.append("")
    report.append("最后10个epoch平均 (稳定性):")
    report.append(f"  {'指标':<15} {'M1基线':>10} {'M2+HA-WFE':>10} {'差异':>10}")
    report.append(f"  {'─'*45}")
    report.append(f"  {'avg PSNR':<15} {m1_avg_psnr:>10.2f} {m2_avg_psnr:>10.2f} {m2_avg_psnr-m1_avg_psnr:>+10.2f}")
    report.append(f"  {'avg SSIM':<15} {m1_avg_ssim:>10.4f} {m2_avg_ssim:>10.4f} {m2_avg_ssim-m1_avg_ssim:>+10.4f}")
    report.append("")
    
    if hawfe_params:
        report.append("HA-WFE参数变化 (零初始化→学习值):")
        report.append(f"  初始: alpha_ll=0, alpha_hf=0, beta=0")
        final = hawfe_params[-1]
        report.append(f"  最终: alpha_ll={final['a_ll']:.4f}, alpha_hf={final['a_hf']:.4f}, beta={final['beta']:.4f}")
        if final['beta'] > 0.001:
            report.append(f"  → HA-WFE已被激活 (beta从0增长到{final['beta']:.4f})")
        else:
            report.append(f"  → HA-WFE未被充分激活 (beta={final['beta']:.4f}, 可能需要更多epoch)")
        report.append("")
    
    # 结论
    diff_psnr = m2_best['psnr'] - m1_best['psnr']
    report.append("结论:")
    if diff_psnr > 0.3:
        report.append(f"  ✓ HA-WFE带来显著提升: PSNR +{diff_psnr:.2f}dB")
        report.append(f"  → 继续Phase 3 (HazeCLIP蒸馏)")
    elif diff_psnr > 0:
        report.append(f"  △ HA-WFE带来微弱提升: PSNR +{diff_psnr:.2f}dB")
        report.append(f"  → 可考虑调整HA-WFE位置或子带策略后再试")
    else:
        report.append(f"  ✗ HA-WFE未带来提升: PSNR {diff_psnr:.2f}dB")
        report.append(f"  → 需要排查原因 (数据量? 训练策略? 模块位置?)")
    
    report.append("")
    report.append(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_text = "\n".join(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text, flush=True)
    write_status("全部完成! 报告已保存到: " + REPORT_FILE)


if __name__ == "__main__":
    # 从M1日志获取训练配置
    config_epochs = 100
    
    # Step 1: 等待M1完成
    wait_for_m1()
    
    # 给M1留几秒保存最终checkpoint
    time.sleep(5)
    
    # Step 2: 启动M2
    run_m2()
    
    # Step 3: 生成对比报告
    generate_report()
    
    # 在状态文件里写完成标记
    write_status("全部完成! 查看报告: " + REPORT_FILE)
