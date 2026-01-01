
import torch
import torch.nn.functional as F
import numpy as np
import sys
from torch import nn
import math


def sa_decode_4d(codes, C):
    """
    Reconstruct the vectors from the codebook C using the indices in codes.

    C is shaped (num_heads, M, c, d//M) # different codebooks for each head
            or  (M, c, d//M) # same codebook for all heads

    codes is shaped (bs, num_heads, n, M)

    returns (bs, num_heads, n, d)
    """
    bs, num_heads, n, M = codes.shape
    
    if len(C.shape) == 3:
        _, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)  # (bs, num_heads, 1, c, d_m)

    codes = codes.to(torch.long)
    # Expand codes to (bs, num_heads, n, M, 1)
    codes_expanded = codes.unsqueeze(-1)  # (bs, num_heads, n, M, 1)

    # Expand codes to match d_m dimension
    codes_expanded = codes_expanded.expand(-1, -1, -1, -1, d_m)  # (bs, num_heads, n, M, d_m)

    # Expand C to match n dimension
    C_expanded = C.unsqueeze(2).expand(-1, -1, n, M, -1, -1)  # (bs, num_heads, n, 1, c, d_m)

    # Gather codewords
    
    decoded = torch.gather(C_expanded, 4, codes_expanded.unsqueeze(-2))  
    
    decoded = decoded.squeeze(4)  # (bs, num_heads, n, M, d_m)

    # Reshape decoded to (bs, num_heads, n, M * d_m)
    decoded = decoded.reshape(bs, num_heads, n, M * d_m)

    return decoded


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def flash_decoding_reference_implementation(
    query: torch.Tensor,
    key_codes: torch.Tensor, 
    value_codes: torch.Tensor,
    key_cents: torch.Tensor,
    value_cents: torch.Tensor,
    key_residuals: torch.Tensor,
    value_residuals: torch.Tensor,
    r: int
) -> torch.Tensor:

    bs, nh, _, d = query.shape
    nh_k = key_codes.shape[1]
    seq_len = key_codes.shape[2]
    M_key = key_codes.shape[3]
    M_value = value_codes.shape[3]
    C = key_cents.shape[1]
    
    print(f"  🔢 参考实现参数: bs={bs}, nh={nh}, nh_k={nh_k}, seq_len={seq_len}")
    print(f"      M_key={M_key}, M_value={M_value}, C={C}, r={r}")
    
    # 1. 重构key和value
    # 从码本重构keys
    sa_decode_4d_keys = sa_decode_4d(key_codes, key_cents)
    sa_decode_4d_values = sa_decode_4d(value_codes, value_cents)

    print(f"  📊 重构后形状:")
    print(f"      sa_decode_4d_keys: {sa_decode_4d_keys.shape}")
    print(f"      sa_decode_4d_values: {sa_decode_4d_values.shape}")
    print(f"      key_residuals: {key_residuals.shape}")
    print(f"      value_residuals: {value_residuals.shape}")

    reconstructed_keys = torch.cat([sa_decode_4d_keys, key_residuals[:, :, :r, :]], dim=2)
    reconstructed_values = torch.cat([sa_decode_4d_values, value_residuals[:, :, :r, :]], dim=2)
    
    print(f"  📊 拼接后形状:")
    print(f"      reconstructed_keys: {reconstructed_keys.shape}")
    print(f"      reconstructed_values: {reconstructed_values.shape}")
    
    # 如果nh != nh_k，需要处理组查询注意力
    if nh != nh_k:
        # 假设是分组注意力，扩展key和value
        group_size = nh // nh_k
        reconstructed_keys = repeat_kv(reconstructed_keys, group_size)
        reconstructed_values = repeat_kv(reconstructed_values, group_size)
    
    causal_mask = None
    is_causal = True if causal_mask is None and query.shape[2] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        reconstructed_keys,
        reconstructed_values,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    return attn_output

def compare_outputs(cuda_output: torch.Tensor, ref_output: torch.Tensor, tolerance: float = 1e-2) -> bool:
    print("🔍 详细对比CUDA输出和参考实现...")
    
    # 1. 基本检查
    if cuda_output.shape != ref_output.shape:
        print(f"❌ 形状不匹配: CUDA={cuda_output.shape}, 参考={ref_output.shape}")
        return False

    # 3. 统计对比
    print(f"📈 统计对比:")
    print(f"   CUDA - min:{cuda_output.min():.6f}, max:{cuda_output.max():.6f}, mean:{cuda_output.mean():.6f}")
    print(f"   参考 - min:{ref_output.min():.6f}, max:{ref_output.max():.6f}, mean:{ref_output.mean():.6f}")
    
    # 4. 绝对值平均值
    error = (cuda_output-ref_output).abs().mean().item()
    print(f"   abs_mean_ratio: {error:.4f}")
    return error

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

def run_single_test(test_params, device, bindings, test_idx):
    """运行单个测试点"""
    bs, nh, nh_k, seq_len, d, M_key, M_value, C, Lt, r = test_params
    Ns = l2Ns(seq_len)
    
    print(f"\n{'='*80}")
    print(f"🧪 测试点 {test_idx}: bs={bs}, nh={nh}, nh_k={nh_k}, seq_len={seq_len}")
    print(f"   Ns={Ns}, d={d}, M_key={M_key}, M_value={M_value}, C={C}, Lt={Lt}, r={r}")
    print(f"{'='*80}")
    
    # 使用固定种子确保可重复性
    torch.manual_seed(42 + test_idx)
    
    print("\n📦 生成测试数据...")
    # 生成测试数据
    query = torch.randn(bs, nh, 1, d, dtype=torch.float16, device=device)
    key_codes = torch.randint(0, C, (bs, nh_k, seq_len, M_key), dtype=torch.uint8, device=device)
    value_codes = torch.randint(0, C, (bs, nh_k, seq_len, M_value), dtype=torch.uint8, device=device)
    key_cents = torch.randn(1, C, d // M_key, dtype=torch.float16, device=device)
    value_cents = torch.randn(1, C, d // M_value, dtype=torch.float16, device=device)
    key_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)
    value_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)
    
    key_residuals[:, :, r:, :] = 0
    value_residuals[:, :, r:, :] = 0
    
    print("✅ 测试数据生成完成")
    
    # 获取CUDA函数
    func_name = f"flash_decoding_allocated_buffer_f16u8_Ns{Ns}Lt{Lt}d{d}M_key{M_key}M_value{M_value}C{C}"
    
    if not hasattr(bindings, func_name):
        print(f"❌ 函数不存在: {func_name}")
        return False, 0.0
    
    func = getattr(bindings, func_name)
    print(f"✅ 获取CUDA函数: {func_name}")
    
    print("\n" + "-"*60)
    print("⚡ 执行CUDA实现...")
    print("-"*60)
    
    # 执行CUDA版本
    partial_out_buffer = torch.zeros(bs, nh, Ns + 1, d, dtype=torch.float16, device=device)
    partial_lse_buffer = torch.zeros(bs, nh, Ns + 1, dtype=torch.float16, device=device)
    
    try:
        # 打印输入参数的形状
        print(f"📋 输入参数形状:")
        print(f"   query: {query.shape}")
        print(f"   key_codes: {key_codes.shape}")
        print(f"   value_codes: {value_codes.shape}")
        print(f"   key_cents: {key_cents.shape}")
        print(f"   value_cents: {value_cents.shape}")
        print(f"   key_residuals: {key_residuals.shape}")
        print(f"   value_residuals: {value_residuals.shape}")
        
        cuda_result = func(
            query, key_codes, value_codes,
            key_cents, value_cents,
            key_residuals, value_residuals,
            r, partial_out_buffer, partial_lse_buffer
        )
        print("✅ CUDA实现执行成功")
        print(f"📊 CUDA输出形状: {cuda_result.shape}")
        print(f"📈 CUDA输出统计: min={cuda_result.min():.6f}, max={cuda_result.max():.6f}, mean={cuda_result.mean():.6f}")
        
    except Exception as e:
        print(f"❌ CUDA实现执行失败: {e}")
        return False, 0.0
    
    print("\n" + "-"*60)
    print("🔢 执行参考实现...")
    print("-"*60)
    
    # 执行参考实现
    try:
        ref_result = flash_decoding_reference_implementation(
            query, key_codes, value_codes,
            key_cents, value_cents,
            key_residuals, value_residuals, r
        )
        print("✅ 参考实现执行成功")
        print(f"📊 参考输出形状: {ref_result.shape}")
        print(f"📈 参考输出统计: min={ref_result.min():.6f}, max={ref_result.max():.6f}, mean={ref_result.mean():.6f}")
        
    except Exception as e:
        print(f"❌ 参考实现执行失败: {e}")
        print(f"错误详情: {e}")
        return False, 0.0
    
    print("\n" + "-"*60)
    print("🔍 对比验证...")
    print("-"*60)
    
    error = compare_outputs(cuda_result, ref_result, tolerance=1e-2)
    
    # 判断是否通过测试
    passed = error < 1e-2
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} 测试点 {test_idx}: 误差 = {error:.6f}")
    
    return passed, error

def test_correctness():
    """测试CUDA实现的正确性"""
    print("🧪 Flash Decoding CUDA实现正确性测试")
    print("=" * 80)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return False
    
    device = torch.device("cuda:0")
    print(f"🚀 使用设备: {device}")
    
    # 导入模块
    try:
        import bindings
        print("✅ 成功导入bindings模块")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 定义所有测试点 (bs, nh, nh_k, len, d, M_key, M_value, C, Lt, r)
    test_cases = [
        # 测试点1
        (1, 32, 8, 98304, 128, 32, 32, 256, 128, 32),
        
        # 测试点2
        (1, 32, 8, 196608, 128, 32, 32, 256, 128, 64),
        
        # 测试点3
        (1, 32, 8, 111, 128, 32, 32, 256, 128, 1),
        
        # 测试点4
        (16, 32, 8, 16384, 128, 32, 32, 256, 128, 16),
        
        # 测试点5
        (16, 32, 8, 16384, 64, 32, 32, 256, 64, 16),
        
        # 测试点6
        (8, 32, 8, 16384, 64, 16, 16, 256, 64, 16),
        
        # 测试点7
        (16, 32, 8, 16384, 64, 64, 64, 256, 64, 16),
        
        # 测试点8
        (4, 32, 8, 16384, 128, 16, 16, 256, 128, 64),
        
        # 测试点9
        (4, 32, 32, 16384, 128, 16, 16, 256, 128, 8),
    ]
    
    print(f"📊 总共 {len(test_cases)} 个测试点")
    
    # 统计结果
    passed_tests = 0
    failed_tests = 0
    errors = []
    
    # 循环执行所有测试
    for i, test_params in enumerate(test_cases, 1):
        passed, error = run_single_test(test_params, device, bindings, i)
        
        if passed:
            passed_tests += 1
        else:
            failed_tests += 1
        
        errors.append(error)

        print("\n" + "🔄" * 40)
    
    # 输出总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    print(f"✅ 通过: {passed_tests}/{len(test_cases)} ({passed_tests/len(test_cases)*100:.1f}%)")
    print(f"❌ 失败: {failed_tests}/{len(test_cases)} ({failed_tests/len(test_cases)*100:.1f}%)")
    
    if errors:
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        print(f"📈 误差统计:")
        print(f"   平均误差: {avg_error:.6f}")
        print(f"   最大误差: {max_error:.6f}")
        print(f"   最小误差: {min_error:.6f}")
    
    # 列出失败的测试点
    if failed_tests > 0:
        print(f"\n❌ 失败的测试点:")
        for i, (test_params, error) in enumerate(zip(test_cases, errors), 1):
            if error >= 1e-2:
                bs, nh, nh_k, seq_len, d, M_key, M_value, C, Lt, r = test_params
                print(f"   测试点 {i}: bs={bs}, nh={nh}, nh_k={nh_k}, seq_len={seq_len}, 误差={error:.6f}")
    
    overall_success = failed_tests == 0
    status = "🎉 全部测试通过!" if overall_success else f"⚠️  有 {failed_tests} 个测试失败"
    print(f"\n{status}")
    
    return overall_success

if __name__ == "__main__":
    success = test_correctness()