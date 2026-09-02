"""
监控M3训练进度, 完成后自动运行四模型对比
"""
import os, sys, time, glob

LOG_FILE = r"C:\Users\245702~1\AppData\Local\Temp\trae-agent-toolhost\jobs\job-5b4ad6d0bbcd496a88a09a72abbced45\output.log"
M3_CKPT = r"D:\dehaze_fusion\icewave_output\m3_clip_distill\checkpoints\m3_best.pth"
COMPARE_SCRIPT = r"c:\Users\2457025871\.trae-cn\work\6a7018ff8440e8370e6f184d\pipeline_m3_compare.py"

def get_last_epoch():
    if not os.path.exists(LOG_FILE):
        return 0, ""
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "Epoch" in line and "loss=" in line:
            return 1, line.strip()
    return 0, ""

def check_complete():
    if not os.path.exists(LOG_FILE):
        return False
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return "训练完成" in content or "Phase 3" in content and "best PSNR" in content

print("监控M3训练进度...")
print(f"日志: {LOG_FILE}")
print(f"检查点: {M3_CKPT}")
print()

while True:
    found, last_line = get_last_epoch()
    if found:
        print(f"[{time.strftime('%H:%M:%S')}] {last_line}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 等待训练输出...")

    if check_complete():
        print("\n>>> 训练完成! <<<")
        if os.path.exists(M3_CKPT):
            print(f"M3最佳检查点已保存: {M3_CKPT}")
            print("运行四模型对比...")
            os.system(f'python "{COMPARE_SCRIPT}"')
        else:
            print("M3检查点未找到, 请检查训练是否成功")
        break

    if os.path.exists(M3_CKPT):
        mtime = os.path.getmtime(M3_CKPT)
        age = time.time() - mtime
        if age > 600:
            print(f"\n检查点已 {age/60:.0f} 分钟未更新, 训练可能已完成或停止")
            print("尝试运行对比...")
            os.system(f'python "{COMPARE_SCRIPT}"')
            break

    time.sleep(120)
