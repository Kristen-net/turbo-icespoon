"""
性能基准测试脚本
=================
测试三赛道 + 融合在 RTX 5060 上的推理时间、显存、吞吐量

使用方法:
    python benchmark.py --input test_hazy.png --iterations 10
    python benchmark.py --input test_hazy.png --resolutions 480 720 1080
"""

import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from fusion_inference import (
        load_image, save_image, tile_inference,
        HazeCLIPEngine, DiffDehazeEngine, WDMambaEngine,
        FusionEngine
    )
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


def benchmark_single(engine, img, name, iterations=10, fp16=True, **kwargs):
    """基准测试单个引擎"""
    print(f"\n[{name}] 基准测试 ({iterations} 次迭代)...")
    print("-" * 50)

    # 预热
    print("  预热中...")
    for _ in range(3):
        torch.cuda.empty_cache()
        try:
            if hasattr(engine, 'infer'):
                if name == 'DiffDehaze':
                    _, _, _ = engine.infer(img, fp16=fp16, tile_size=256, overlap=32)
                else:
                    _, _, _ = engine.infer(img, fp16=fp16)
        except Exception as e:
            print(f"  预热失败: {e}")
            return None

    # 正式测试
    times = []
    vrams = []

    for i in range(iterations):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        try:
            if hasattr(engine, 'infer'):
                if name == 'DiffDehaze':
                    out, _, vram = engine.infer(img, fp16=fp16, tile_size=256, overlap=32)
                else:
                    out, _, vram = engine.infer(img, fp16=fp16)
            elapsed = time.time() - start
            times.append(elapsed)
            vrams.append(vram)
            print(f"  迭代 {i+1}/{iterations}: {elapsed:.3f}s, VRAM={vram:.2f}GB")
        except Exception as e:
            print(f"  迭代 {i+1} 失败: {e}")
            times.append(float('inf'))
            vrams.append(0)

    # 统计
    times = np.array(times)
    vrams = np.array(vrams)

    stats = {
        'name': name,
        'iterations': iterations,
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'time_min': float(np.min(times)),
        'time_max': float(np.max(times)),
        'time_p50': float(np.percentile(times, 50)),
        'time_p95': float(np.percentile(times, 95)),
        'vram_mean': float(np.mean(vrams)),
        'vram_max': float(np.max(vrams)),
        'throughput_fps': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0,
    }

    print(f"\n  [{name}] 统计:")
    print(f"    平均时间: {stats['time_mean']:.3f}s ± {stats['time_std']:.3f}s")
    print(f"    最小/最大: {stats['time_min']:.3f}s / {stats['time_max']:.3f}s")
    print(f"    P50/P95: {stats['time_p50']:.3f}s / {stats['time_p95']:.3f}s")
    print(f"    峰值显存: {stats['vram_mean']:.2f} GB (max: {stats['vram_max']:.2f} GB)")
    print(f"    吞吐量: {stats['throughput_fps']:.2f} FPS")

    return stats


def benchmark_resolution(engine, img_path, name, resolutions, iterations=5, fp16=True):
    """测试不同分辨率下的性能"""
    print(f"\n[{name}] 分辨率基准测试...")
    print("-" * 50)

    results = []
    for res in resolutions:
        # 读取并调整分辨率
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if res == 'original':
            target_h, target_w = h, w
        else:
            scale = res / max(h, w)
            target_h, target_w = int(h * scale), int(w * scale)

        img = cv2.resize(img, (target_w, target_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

        print(f"\n  分辨率: {target_w}x{target_h}")

        # 预热
        try:
            if name == 'DiffDehaze':
                engine.infer(img_tensor, fp16=fp16, tile_size=256, overlap=32)
            else:
                engine.infer(img_tensor, fp16=fp16)
        except:
            pass

        # 测试
        times = []
        vrams = []
        for _ in range(iterations):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            try:
                if name == 'DiffDehaze':
                    _, _, vram = engine.infer(img_tensor, fp16=fp16, tile_size=256, overlap=32)
                else:
                    _, _, vram = engine.infer(img_tensor, fp16=fp16)
                elapsed = time.time() - start
                times.append(elapsed)
                vrams.append(vram)
            except Exception as e:
                print(f"    失败: {e}")
                times.append(float('inf'))
                vrams.append(0)

        avg_time = np.mean(times)
        avg_vram = np.mean(vrams)
        print(f"    平均: {avg_time:.3f}s, VRAM: {avg_vram:.2f}GB")

        results.append({
            'resolution': f"{target_w}x{target_h}",
            'time_mean': float(avg_time),
            'vram_mean': float(avg_vram),
            'throughput_fps': float(1.0 / avg_time) if avg_time > 0 else 0,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="性能基准测试")
    parser.add_argument('--input', type=str, required=True, help='输入图片路径')
    parser.add_argument('--iterations', type=int, default=10, help='每项测试迭代次数')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[480, 720, 1080],
                        help='测试分辨率列表')
    parser.add_argument('--hazeclip-weights', type=str, default='')
    parser.add_argument('--diffdehaze-weights', type=str, default='')
    parser.add_argument('--wdmamba-weights', type=str, default='')
    parser.add_argument('--output', type=str, default='benchmark_report.json')

    args = parser.parse_args()

    print("=" * 60)
    print("RTX 5060 性能基准测试")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.2f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"输入: {args.input}")
    print(f"迭代次数: {args.iterations}")
    print()

    if not FUSION_AVAILABLE:
        print("[错误] 无法导入 fusion_inference，请确保文件在同一目录")
        sys.exit(1)

    # 加载图片
    img = load_image(args.input)
    print(f"图像尺寸: {img.shape}")
    print()

    all_results = {}

    # 测试三赛道
    print("=" * 60)
    print("Phase 1: 单赛道基准测试")
    print("=" * 60)

    hazeclip = HazeCLIPEngine(args.hazeclip_weights)
    diffdehaze = DiffDehazeEngine(args.diffdehaze_weights, num_steps=15)
    wdmamba = WDMambaEngine(args.wdmamba_weights)

    all_results['hazeclip'] = benchmark_single(hazeclip, img, 'HazeCLIP', args.iterations)
    all_results['diffdehaze'] = benchmark_single(diffdehaze, img, 'DiffDehaze', args.iterations)
    all_results['wdmamba'] = benchmark_single(wdmamba, img, 'WDMamba', args.iterations)

    # 测试融合
    print("\n" + "=" * 60)
    print("Phase 2: 融合基准测试")
    print("=" * 60)

    fusion = FusionEngine(hazeclip, diffdehaze, wdmamba)

    for level in [1, 2, 3, 4]:
        print(f"\n[融合 Level {level}] 基准测试 ({args.iterations} 次迭代)...")
        print("-" * 50)

        times = []
        vrams = []

        # 预热
        for _ in range(2):
            torch.cuda.empty_cache()
            try:
                fusion.infer(img, fusion_level=level, fp16=True)
            except:
                pass

        for i in range(args.iterations):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            try:
                _, _, vram = fusion.infer(img, fusion_level=level, fp16=True)
                elapsed = time.time() - start
                times.append(elapsed)
                vrams.append(vram)
                print(f"  迭代 {i+1}/{args.iterations}: {elapsed:.3f}s, VRAM={vram:.2f}GB")
            except Exception as e:
                print(f"  迭代 {i+1} 失败: {e}")
                times.append(float('inf'))
                vrams.append(0)

        times = np.array(times)
        vrams = np.array(vrams)
        key = f'fusion_level_{level}'
        all_results[key] = {
            'name': f'融合 Level {level}',
            'iterations': args.iterations,
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'time_min': float(np.min(times)),
            'time_max': float(np.max(times)),
            'time_p50': float(np.percentile(times, 50)),
            'time_p95': float(np.percentile(times, 95)),
            'vram_mean': float(np.mean(vrams)),
            'vram_max': float(np.max(vrams)),
            'throughput_fps': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0,
        }

        print(f"\n  [融合 Level {level}] 统计:")
        print(f"    平均时间: {all_results[key]['time_mean']:.3f}s ± {all_results[key]['time_std']:.3f}s")
        print(f"    峰值显存: {all_results[key]['vram_mean']:.2f} GB (max: {all_results[key]['vram_max']:.2f} GB)")
        print(f"    吞吐量: {all_results[key]['throughput_fps']:.2f} FPS")

    # 测试不同分辨率
    print("\n" + "=" * 60)
    print("Phase 3: 分辨率基准测试")
    print("=" * 60)

    resolution_results = {}
    for name, engine in [('HazeCLIP', hazeclip), ('WDMamba', wdmamba)]:
        resolution_results[name] = benchmark_resolution(
            engine, args.input, name, args.resolutions, iterations=3
        )

    all_results['resolution_test'] = resolution_results

    # 汇总表
    print("\n" + "=" * 60)
    print("基准测试汇总")
    print("=" * 60)
    print(f"{'方法':<20} {'平均时间(s)':<15} {'峰值显存(GB)':<15} {'吞吐量(FPS)':<15}")
    print("-" * 65)

    for key, stats in all_results.items():
        if key == 'resolution_test':
            continue
        if stats:
            print(f"{stats.get('name', key):<20} {stats['time_mean']:<15.3f} {stats['vram_mean']:<15.2f} {stats['throughput_fps']:<15.2f}")

    print("-" * 65)
    print()

    # 保存报告
    report = {
        'gpu': torch.cuda.get_device_name(0),
        'vram_total_gb': torch.cuda.get_device_properties(0).total_mem / 1024**3,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'input': args.input,
        'iterations': args.iterations,
        'results': all_results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"报告已保存: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
