"""监控M4训练进度, 完成后自动运行五模型对比评测"""

import os
import time
import subprocess

LOG_FILE = r"C:\Users\245702~1\AppData\Local\Temp\trae-agent-toolhost\jobs\job-8193cd9d82594dd9832020c4ed8ce917\output.log"
M4_CKPT = r"D:\dehaze_fusion\icewave_output\m4_itl\checkpoints\m4_best.pth"
COMPARE_SCRIPT = r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d\pipeline_m4_compare.py"
PYTHON_EXE = r"C:\Users\2457025871\miniconda3\envs\dehaze_fusion\python.exe"

print("监控M4训练进度...")
print(f"日志: {LOG_FILE}")
print(f"检查点: {M4_CKPT}")
print()

last_mtime = 0
no_update_count = 0

while True:
    # 检查训练是否完成
    complete = False
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if "训练完成" in content or "Phase 4" in content and "best PSNR" in content:
            complete = True

        # 打印最新epoch行
        lines = content.strip().split('\n')
        epoch_lines = [l for l in lines if l.startswith("Epoch")]
        if epoch_lines:
            last_line = epoch_lines[-1]
            # 限制打印频率
            mtime = os.path.getmtime(LOG_FILE)
            if mtime != last_mtime:
                print(f"[{time.strftime('%H:%M:%S')}] {last_line}")
                last_mtime = mtime

    # 检查点未更新检测
    if os.path.exists(M4_CKPT):
        ckpt_mtime = os.path.getmtime(M4_CKPT)
        if ckpt_mtime == last_mtime:
            no_update_count += 1
        else:
            no_update_count = 0
            last_mtime = ckpt_mtime

        if no_update_count > 5:
            print(f"\n检查点已 {no_update_count * 2} 分钟未更新, 训练可能已完成或停止")
            complete = True

    if complete:
        print("\n>>> 训练完成! <<<")
        if os.path.exists(M4_CKPT):
            print("运行五模型对比评测...")
            subprocess.run([PYTHON_EXE, "-u", COMPARE_SCRIPT], check=True)
        else:
            print("警告: 最佳检查点不存在!")
        break

    time.sleep(120)
