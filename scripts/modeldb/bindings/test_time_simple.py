#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flash Decoding 简化时间测试脚本

快速测试CUDA实现的时间性能 - 多序列长度版本
"""

import torch
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def measure_time(func, *args, num_runs=20, warmup=5, print_each_time=True):
    """简单的时间+显存测量函数"""
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    # 预热
    print(f"🔥 开始预热 {warmup} 次...")
    for i in range(warmup):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        func(*args)
    
    print(f"\n⏱️ 开始正式测试 {num_runs} 次...")
    # 测试
    times = []
    for i in range(num_runs):
        time.sleep(0.5)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()

        execution_time = (end - start) * 1000  # 毫秒
        times.append(execution_time)

    # 测试后统计显存
    current_mem = torch.cuda.memory_allocated(device) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    return times, np.mean(times), np.std(times), result, current_mem, peak_mem

def l2Ns(l):
    """
    Heuristic to select proper number of splits Ns for a sequence of length l.
    """
    if l > 8192:
        return 64
    elif l > 2048:
        return 32 
    elif l > 256:
        return 16
    elif l > 128:
        return 4
    elif l > 64:
        return 2
    else:
        return 1 # To small to benefit from parallelization

def test_single_seq_len(seq_len, bindings, device):
    """测试单个序列长度"""
    print(f"\n{'='*60}")
    print(f"🧪 测试序列长度: {seq_len}")
    print(f"{'='*60}")
    
    # 测试参数
    bs, nh, nh_k = 1, 32, 8
    Ns, d, M_key, M_value, C = l2Ns(seq_len), 128, 16, 16, 256
    Lt, r = 128, 0
    
    print(f"📊 测试参数: bs={bs}, nh={nh}, seq_len={seq_len}, Ns={Ns}")
    
    # 固定随机种子
    torch.manual_seed(42)
    
    # 生成测试数据
    query = torch.randn(bs, nh, 1, d, dtype=torch.float16, device=device)
    key_codes = torch.randint(0, C, (bs, nh_k, seq_len, M_key), dtype=torch.uint8, device=device)
    value_codes = torch.randint(0, C, (bs, nh_k, seq_len, M_value), dtype=torch.uint8, device=device)
    key_cents = torch.randn(1, C, d // M_key, dtype=torch.float16, device=device)
    value_cents = torch.randn(1, C, d // M_value, dtype=torch.float16, device=device)
    key_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)
    value_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)
    
    # 获取CUDA函数
    func_name = f"flash_decoding_allocated_buffer_f16u8_Ns{Ns}Lt{Lt}d{d}M_key{M_key}M_value{M_value}C{C}"
    
    if not hasattr(bindings, func_name):
        print(f"❌ 函数不存在: {func_name}")
        return None
    
    func = getattr(bindings, func_name)
    print(f"✅ 找到CUDA函数: {func_name}")
    
    # 分配缓冲区
    partial_out_buffer = torch.zeros(bs, nh, Ns + 1, d, dtype=torch.float16, device=device)
    partial_lse_buffer = torch.zeros(bs, nh, Ns + 1, dtype=torch.float16, device=device)
    
    def cuda_func():
        return func(
            query, key_codes, value_codes,
            key_cents, value_cents,
            key_residuals, value_residuals,
            r, partial_out_buffer, partial_lse_buffer
        )
    
    print(f"\n⏱️ 开始时间测试...")
    print("-" * 50)
    
    try:
        times_list, avg_time, std_time, result, cur_mem, peak_mem = measure_time(
            cuda_func, num_runs=15, warmup=10
        )
        print(f"   所有时间: {[f'{t:.3f}' for t in times_list]}")
        print(f"✅ 测试完成!")
        print(f"📊 平均执行时间: {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"📈 输出形状: {result.shape}")
        print(f"💾 当前显存占用: {cur_mem:.2f} MB")
        print(f"📈 峰值显存占用: {peak_mem:.2f} MB")
        
        return {
            'seq_len': seq_len,
            'avg_time': avg_time,
            'std_time': std_time,
            'times': times_list,
            'current_memory': cur_mem,
            'peak_memory': peak_mem,
            'output_shape': result.shape
        }

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def main():
    print("⚡ Flash Decoding 多序列长度时间测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return
    
    device = torch.device("cuda:0")
    print(f"🚀 使用设备: {device}")
    
    # 导入模块
    try:
        import bindings
        print("✅ 成功导入bindings模块")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 测试不同序列长度
    seq_lengths = [98304, 131072, 163840, 196608]
    results = []
    
    print(f"\n🎯 将测试以下序列长度: {seq_lengths}")
    
    for i, seq_len in enumerate(seq_lengths):
        print(f"\n🔄 进度: {i+1}/{len(seq_lengths)}")
        result = test_single_seq_len(seq_len, bindings, device)
        if result:
            results.append(result)
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 如果不是最后一个，稍作休息
        if i < len(seq_lengths) - 1:
            print(f"\n⏳ 休息2秒后继续下一个测试...")
            time.sleep(2)
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("📋 测试汇总结果")
    print(f"{'='*80}")
    
    if results:
        print(f"{'序列长度':<12} {'平均时间(ms)':<15} {'标准差(ms)':<15} {'峰值显存(MB)':<15}")
        print("-" * 70)
        for result in results:
            print(f"{result['seq_len']:<12} {result['avg_time']:<15.3f} "
                  f"{result['std_time']:<15.3f} {result['peak_memory']:<15.2f}")
        
    else:
        print("❌ 没有成功的测试结果")

if __name__ == "__main__":
    main()