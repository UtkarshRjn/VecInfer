import sys
import os
import torch

# 将 build 目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import bindings

def l2Ns(l):
    """
    Heuristic to select proper number of splits Ns for a sequence of length l.
    """
    if l > 8192:
        return 64
    elif l > 2048:
        return 32 # Ns32 is manually selected. It's the fastest kernel for long sequences with >2k tokens
        # we are not benefiting from bigger Ns possibly due to hardware limitations
    elif l > 256:
        return 16
    elif l > 128:
        return 4
    elif l > 64:
        return 2
    else:
        return 1 # To small to benefit from parallelization

bs, nh, nh_k, len = 1, 32, 8, 98304
Ns, d, M_key, M_value, C = l2Ns(len), 128, 16, 16, 256
Lt, r = 128, 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

torch.manual_seed(42)  

print("生成测试数据...")
print(f"参数: bs={bs}, nh={nh}, nh_k={nh_k}, len={len}")
print(f"      Ns={Ns}, d={d}, M_key={M_key}, M_value={M_value}, C={C}, Lt={Lt}")

query = torch.randn(bs, nh, 1, d, dtype=torch.float16, device=device)
key_codes = torch.randint(0, C, (bs, nh_k, len, M_key), dtype=torch.uint8, device=device)
value_codes = torch.randint(0, C, (bs, nh_k, len, M_value), dtype=torch.uint8, device=device)
key_cents = torch.randn(1, C, d // M_key, dtype=torch.float16, device=device)
value_cents = torch.randn(1, C, d // M_value, dtype=torch.float16, device=device)
key_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)
value_residuals = torch.randn(bs, nh_k, Lt, d, dtype=torch.float16, device=device)

key_residuals[:, :, r:, :] = 0
value_residuals[:, :, r:, :] = 0

# 输出缓冲区
partial_out_buffer = torch.zeros(bs, nh, Ns + 1, d, dtype=torch.float16, device=device)
partial_lse_buffer = torch.zeros(bs, nh, Ns + 1, dtype=torch.float16, device=device)

print("输入张量形状检查:")
print(f"  query: {query.shape}")
print(f"  key_codes: {key_codes.shape}")
print(f"  value_codes: {value_codes.shape}")
print(f"  key_cents: {key_cents.shape}")
print(f"  value_cents: {value_cents.shape}")
print(f"  key_residuals: {key_residuals.shape}")
print(f"  value_residuals: {value_residuals.shape}")
print(f"  partial_out_buffer: {partial_out_buffer.shape}")
print(f"  partial_lse_buffer: {partial_lse_buffer.shape}")

print("调用 flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M_key32M_value32C256...")

result = bindings.flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M_key16M_value16C256(
    query,               
    key_codes,          
    value_codes,         
    key_cents,          
    value_cents,         
    key_residuals,       
    value_residuals,     
    r,                  
    partial_out_buffer,  
    partial_lse_buffer   
)

print(f"✅ 成功执行! 结果形状: {result.shape}")
