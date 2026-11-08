import torch
import os
from pykeops.torch import LazyTensor
from ..utils.Singleton import Singleton

from ..utils.Namespace import UniConfig
from ..utils.Timer import Timer
from transformers.models.llama.modeling_llama import repeat_kv
from torch.nn.functional import scaled_dot_product_attention as sdpa
from fast_hadamard_transform import hadamard_transform
import math


def l2Ns(l):
    """
    Heuristic to select proper number of splits Ns for a sequence of length l.
    """
    if l > 2048:
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
    
def scalarTypeToStr(scalar_t):
    if scalar_t == torch.float32:
        return 'f32'
    elif scalar_t == torch.float16:
        return 'f16'
    else:
        raise ValueError(f"Unknown scalar type: {scalar_t}")

class KernelRegistry:
    def __init__(self, *, M_key=64, M_value=64, d=128, nbits=8, nh=32, scalar_t=torch.float32):
        self.kernels = {}
        self.partial_out_buffers = {}
        self.partial_lse_buffers = {}
        self.M_key = M_key
        self.M_value = M_value
        self.d = d
        self.nbits = nbits
        self.nh = nh
        self.scalar_t = scalar_t

    def get_kernel(self, l=4096):
        Ns = l2Ns(l)
        if Ns not in self.kernels:
            self.kernels[Ns] = self.get_custom_kernel_with_allocated_buffer(l)
        return self.kernels[Ns]
    
    def get_custom_kernel_with_allocated_buffer(self, l=4096):
        M_key = self.M_key
        M_value = self.M_value
        d = self.d
        nbits = self.nbits
        nh = self.nh
        scalar_t = self.scalar_t

        bs = 1 # typical for inference

        if nbits != 8:
            raise NotImplementedError("Only uint8 code type is supported for now")
        
        Ns = l2Ns(l)
        scalar_str = scalarTypeToStr(scalar_t)
        try:
            fname = f"flash_decoding_allocated_buffer_{scalar_str}u8_Ns{Ns}Lt{d}d{d}M_key{M_key}M_value{M_value}C{2**nbits}"
            func = getattr(__import__('bindings'), fname)
        except Exception as e:
            print(f"{fname} not compiled. Modify scripts/modeldb/bindings/setup.py to compile it.")
            raise e

        # prepare allocated buffer, don't worry this is small
        # NOTE: This overhead is not introduced by PQ! It's introduced by parallelizing softmax across blocks
        # i.e. the overhead is implementation-specific
        partial_out_buffer = torch.empty(bs, nh, Ns+1, d, dtype=scalar_t, device='cuda')
        partial_lse_buffer = torch.empty(bs, nh, Ns+1, dtype=scalar_t, device='cuda')

        self.partial_out_buffers[Ns] = partial_out_buffer
        self.partial_lse_buffers[Ns] = partial_lse_buffer

        # define a closure so we do not need to specify the buffer every time
        def flash_decoding(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r):
            # partial_lse_buffer.fill_(float('-inf')) # no need to re-initialize
            # partial_out_buffer.fill_(0)
            return func(
                query, 
                key_codes, 
                value_codes, 
                key_cents, 
                value_cents, 
                key_residuals,
                value_residuals,
                r,
                partial_out_buffer, 
                partial_lse_buffer,
            )
        
        return flash_decoding

class DynamicPQCache(metaclass=Singleton):
    def __init__(self, *, bs, nh, num_key_value_heads, M_key, M_value, layer_num, dtype_key=torch.uint8, dtype_value=torch.uint8, nbits_key=8, nbits_value=8, d=128, scalar_t=torch.float32, scales_path=None, sink_tokens):
        self.bs = bs
        self.nh = nh
        self.num_key_value_heads = num_key_value_heads
        self.M_key = M_key
        self.M_value = M_value
        self.layer_num = layer_num
        self.dtype_key = dtype_key
        self.dtype_value = dtype_value
        self.nbits_key = nbits_key
        self.nbits_value = nbits_value
        self.d = d
        self.scalar_t = scalar_t
        self.num_key_value_groups = nh//num_key_value_heads

        self.max_residual_length = d # Common practice: Lt = d to make residual cache square so kernel is easy to implement
        self.registery = KernelRegistry(M_key=M_key, M_value=M_value, d=d, nbits=nbits_key, nh=nh, scalar_t=scalar_t)
        self.scales_path = scales_path
        self.sink_tokens = sink_tokens

        self.init_cache()

    def init_cache(self):
        self.key_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M_key), dtype=self.dtype_key, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.value_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M_value), dtype=self.dtype_value, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.key_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, self.max_residual_length, self.d), dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.value_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, self.max_residual_length, self.d), dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.seen_tokens = [0 for _ in range(self.layer_num)]
        self.residualed_tokens = [0 for _ in range(self.layer_num)]
       
        self.scales = torch.load(self.scales_path, map_location=torch.device('cuda'))
        self.full_key_cache = [(self.bs, self.num_key_value_heads, 0, self.d) for _ in range(self.layer_num)]
        self.full_value_cache = [(self.bs, self.num_key_value_heads, 0, self.d) for _ in range(self.layer_num)]

    def cat_codes(self, key_codes, value_codes, layer_idx):
        """
        Dynamic cache's concatenate is very inefficient. We optimize cat by triggering it only when residual cache is full.
        """
        # naive cat
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
        

    def set_cent(self, key_cents, value_cents):
        """
        cent is of shape (M, c, d//M)
        """
        if isinstance(key_cents, list) and isinstance(value_cents, list):
            if not hasattr(self, 'layer_cents'):
                self.key_cent = {}
                self.value_cent = {}
            for i, (key_cent, value_cent) in enumerate(zip(key_cents, value_cents)):     
                if key_cent is value_cent:
                    self._cent = key_cent.contiguous()
                    self.key_cent[i] = self._cent
                    self.value_cent[i] = self._cent
                else:
                    self.key_cent[i] = key_cent.contiguous()
                    self.value_cent[i] = value_cent.contiguous()

    def update(self, key_states, value_states, layer_idx, distort_recent=False):
        """
        When not dispatching to custom kernel, this function is used to update the cache.

        Update the cache and return the updated key-value states. This behaviour is inherently compatible with DynamicCache from transformers.
        """
        # states are of shape (bs, num_key_value_heads, update_length, d)
        # assert cent is set
        assert hasattr(self, 'key_cent') and hasattr(self, 'value_cent')
        self.seen_tokens[layer_idx] += key_states.shape[-2]

        if key_states.shape[-2] != 1:
            key_reserved = key_states[:, :, :self.sink_tokens, :]
            value_reserved = value_states[:, :, :self.sink_tokens, :]
                
            self.key_residual_cache[layer_idx][:, :, :self.sink_tokens, :] = key_reserved
            self.value_residual_cache[layer_idx][:, :, :self.sink_tokens, :] = value_reserved
            self.residualed_tokens[layer_idx] = self.sink_tokens

            remaining_key_states = key_states[:, :, self.sink_tokens:, :]
            remaining_value_states = value_states[:, :, self.sink_tokens:, :]

            # Prefill-like logic for multi-token sequences
            full_chunks = remaining_key_states.shape[-2] // (self.max_residual_length - self.sink_tokens)
            remainder = remaining_key_states.shape[-2] % (self.max_residual_length - self.sink_tokens)

            if full_chunks > 0:
                end_idx = full_chunks * (self.max_residual_length - self.sink_tokens)
                key_full = remaining_key_states[:, :, :end_idx, :]
                value_full = remaining_value_states[:, :, :end_idx, :]
                    
                key_codes = sa_encode_4d_keops(key_full, self.key_cent[layer_idx], self.M_key, target_dtype=self.dtype_key)
                value_codes = sa_encode_4d_keops(value_full, self.value_cent[layer_idx], self.M_value, target_dtype=self.dtype_value)

                self.cat_codes(key_codes, value_codes, layer_idx)

            if remainder > 0:
                start_idx = full_chunks * (self.max_residual_length - self.sink_tokens)
                key_remainder = remaining_key_states[:, :, start_idx:, :]
                value_remainder = remaining_value_states[:, :, start_idx:, :]
                    
                r = self.residualed_tokens[layer_idx]

                self.key_residual_cache[layer_idx][:, :, r:r + remainder, :] = key_remainder
                self.value_residual_cache[layer_idx][:, :, r:r + remainder, :] = value_remainder
                self.residualed_tokens[layer_idx] += remainder

            return key_states, value_states

        else:
            # Decoding-like logic for single token
            r = self.residualed_tokens[layer_idx]
            
            if r == self.max_residual_length:
                key_codes = sa_encode_4d_keops(self.key_residual_cache[layer_idx], self.key_cent[layer_idx], self.M_key, target_dtype=self.dtype_key)
                value_codes = sa_encode_4d_keops(self.value_residual_cache[layer_idx], self.value_cent[layer_idx], self.M_value, target_dtype=self.dtype_value)
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
                self.residualed_tokens[layer_idx] = self.sink_tokens
            
            r = self.residualed_tokens[layer_idx]
            n = key_states.size(2)
            self.key_residual_cache[layer_idx][:, :, r: r + n, :] = key_states
            self.value_residual_cache[layer_idx][:, :, r: r + n, :] = value_states
            self.residualed_tokens[layer_idx] += n

            past_key_states = sa_decode_4d(self.key_cache[layer_idx], self.key_cent[layer_idx])
            past_value_states = sa_decode_4d(self.value_cache[layer_idx], self.value_cent[layer_idx])

            key_states = torch.cat([past_key_states, self.key_residual_cache[layer_idx][:, :, :r + n, :]], dim=2)
            value_states = torch.cat([past_value_states, self.value_residual_cache[layer_idx][:, :, :r + n, :]], dim=2)

            return key_states, value_states
    
    def prefill(self, query_states, key_states, value_states, layer_idx, distort_recent=False):
        self.seen_tokens[layer_idx] += key_states.shape[-2]
        
        with Timer("DynamicPQCache.prefill.scale"):
            key_states.div_(self.scales[f"model.layers.{layer_idx}.self_attn.k_proj"])
            query_states.mul_(repeat_kv(self.scales[f"model.layers.{layer_idx}.self_attn.k_proj"],self.num_key_value_groups))
        
        with Timer("DynamicPQCache.prefill.hadamard"):
            hadamard_dtype = torch.float16
            key_states = hadamard_transform(key_states, scale=1/math.sqrt(key_states.shape[-1])).to(hadamard_dtype) 
            query_states = hadamard_transform(query_states, scale=1/math.sqrt(query_states.shape[-1])).to(hadamard_dtype)  
        
        with Timer("DynamicPQCache.prefill.residual"):
            key_reserved = key_states[:, :, :self.sink_tokens, :]
            value_reserved = value_states[:, :, :self.sink_tokens, :]
                
            self.key_residual_cache[layer_idx][:, :, :self.sink_tokens, :] = key_reserved
            self.value_residual_cache[layer_idx][:, :, :self.sink_tokens, :] = value_reserved
            self.residualed_tokens[layer_idx] = self.sink_tokens

            remaining_key_states = key_states[:, :, self.sink_tokens:, :]
            remaining_value_states = value_states[:, :, self.sink_tokens:, :]

        full_chunks = remaining_key_states.shape[-2] // (self.max_residual_length - self.sink_tokens)
        remainder = remaining_key_states.shape[-2] % (self.max_residual_length - self.sink_tokens)

        if full_chunks > 0:
            with Timer("DynamicPQCache.prefill"):
                with Timer("DynamicPQCache.prefill.encode"):
                    end_idx = full_chunks * (self.max_residual_length - self.sink_tokens)
                    key_full = remaining_key_states[:, :, :end_idx, :]
                    value_full = remaining_value_states[:, :, :end_idx, :]
                    
                    key_codes = sa_encode_4d_keops(key_full, self.key_cent[layer_idx], self.M_key, target_dtype=self.dtype_key)
                    value_codes = sa_encode_4d_keops(value_full, self.value_cent[layer_idx], self.M_value, target_dtype=self.dtype_value)

                with Timer("DynamicPQCache.prefill.cat"):
                    self.cat_codes(key_codes, value_codes, layer_idx)
        
        if remainder > 0:
            with Timer("DynamicPQCache.prefill.residual"):
                start_idx = full_chunks * (self.max_residual_length - self.sink_tokens)
                key_remainder = remaining_key_states[:, :, start_idx:, :]
                value_remainder = remaining_value_states[:, :, start_idx:, :]
                
                r = self.residualed_tokens[layer_idx]
                
                self.key_residual_cache[layer_idx][:, :, r:r + remainder, :] = key_remainder
                self.value_residual_cache[layer_idx][:, :, r:r + remainder, :] = value_remainder
                self.residualed_tokens[layer_idx] += remainder

        with Timer("DynamicPQCache.prefill.attention"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            return sdpa(query_states, key_states, value_states, is_causal=True)
        
    def decoding(self, query_states, key_states, value_states, layer_idx):
        breakdown = UniConfig().breakdown
        r = self.residualed_tokens[layer_idx]      
   
        self.seen_tokens[layer_idx] += key_states.shape[-2]

        with Timer("LlamaSdpaAttention.forward.scale"):
            query_states.mul_(repeat_kv(self.scales[f"model.layers.{layer_idx}.self_attn.k_proj"], self.num_key_value_groups))
            key_states.div_(self.scales[f"model.layers.{layer_idx}.self_attn.k_proj"])
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.hadamard"):
            hadamard_dtype = torch.float16
            key_states = hadamard_transform(key_states, scale=1/math.sqrt(key_states.shape[-1])).to(hadamard_dtype) 
            query_states = hadamard_transform(query_states, scale=1/math.sqrt(query_states.shape[-1])).to(hadamard_dtype)  
            if breakdown is True: torch.cuda.synchronize()

        if r == self.max_residual_length:
            with Timer("LlamaSdpaAttention.forward.flush"):
                with Timer("LlamaSdpaAttention.forward.flush.encode"):
                    key_codes = sa_encode_4d_keops(self.key_residual_cache[layer_idx], self.key_cent[layer_idx], self.M_key, target_dtype=self.dtype_key)
                    value_codes = sa_encode_4d_keops(self.value_residual_cache[layer_idx], self.value_cent[layer_idx], self.M_value, target_dtype=self.dtype_value)
                    if breakdown is True: torch.cuda.synchronize()

                with Timer("LlamaSdpaAttention.forward.flush.cat"):
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
                    if breakdown is True: torch.cuda.synchronize()

                self.residualed_tokens[layer_idx] = self.sink_tokens
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.copy"):
            r = self.residualed_tokens[layer_idx]
            n = key_states.size(2)
            self.key_residual_cache[layer_idx][:, :, r: r + n, :] = key_states
            self.value_residual_cache[layer_idx][:, :, r: r + n, :] = value_states
            self.residualed_tokens[layer_idx] += n
            if breakdown is True: torch.cuda.synchronize()
       
        with Timer("LlamaSdpaAttention.forward.kernel"):
            
            kernel = self.registery.get_kernel(l=self.seen_tokens[layer_idx])
            out = kernel(
                query_states,
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.key_cent[layer_idx],
                self.value_cent[layer_idx],
                self.key_residual_cache[layer_idx],
                self.value_residual_cache[layer_idx],
                self.residualed_tokens[layer_idx]
                )
            if breakdown is True: torch.cuda.synchronize()    

        return out
    

    @property
    def pq_cache_size(self):
        return sum([cache.numel() * cache.element_size() for cache in self.key_cache + self.value_cache])

    @property
    def codebook_size(self):
        if hasattr(self, '_cent'):
            return self._cent.numel() * self._cent.element_size()
        else:
            k_cent_size = self.key_cent.numel() * self.key_cent.element_size()
            v_cent_size = self.value_cent.numel() * self.value_cent.element_size()
            return k_cent_size + v_cent_size

    @property
    def residual_cache_size(self):
        return sum([cache.numel() * cache.element_size() for cache in self.key_residual_cache + self.value_residual_cache])
    
    @property
    def registry_size(self):
        partial_out_size = sum([
            buffer.numel() * buffer.element_size() for buffer in self.registery.partial_out_buffers.values()
        ])
        partial_lse_size = sum([
            buffer.numel() * buffer.element_size() for buffer in self.registery.partial_lse_buffers.values()
        ])
        return partial_out_size + partial_lse_size


def sa_encode_4d_keops(X, C, M, target_dtype=torch.uint8):
    """
    Find the nearest codeword for each vector in X among the rows of C using PyKeOps.
    Returns the indices of the nearest codewords.

    X is shaped (bs, num_heads, n, d) then reshaped to (bs, num_heads, n, M, d//M)
    C is shaped (1, c, d//M) # same codebook for all heads
        or already expanded to (bs, num_heads, M, c, d_m)

    Returns (bs, num_heads, n, M)
    """

    bs, num_heads, n, d = X.shape

    # KeOps illegal-address faults on long sequences (code tasks). The argmin
    # is independent per token, so chunking does not change the codes.
    if n > 2048:
        parts = [
            sa_encode_4d_keops(X[:, :, i:i + 2048].contiguous(), C, M, target_dtype)
            for i in range(0, n, 2048)
        ]
        return torch.cat(parts, dim=2)

    if len(C.shape) == 3:
        _, c, d_m = C.shape
        # Expand C to match the batch structure: (bs, num_heads, M, c, d_m)
        # First add dimensions for M: (1, c, d_m) -> (1, M, c, d_m)
        C = C.unsqueeze(1).expand(-1, M, -1, -1)  # (1, M, c, d_m)
        # Then add batch and head dimensions: (1, M, c, d_m) -> (bs, num_heads, M, c, d_m)
        C = C.unsqueeze(0).expand(bs, num_heads, -1, -1, -1)
    
    # Reshape X to (bs, num_heads, n, M, d_m)
    
    X = X.reshape(bs, num_heads, n, M, d_m)

    # Reshape X and C for batch computation
    # X: (bs * num_heads * M, n, d_m)
    X = X.permute(0, 1, 3, 2, 4).reshape(bs * num_heads * M, n, d_m)
    # C: (bs * num_heads * M, c, d_m)
    C = C.reshape(bs * num_heads * M, -1, d_m)

    # ensure contiguous and convert to float32
    # https://github.com/getkeops/keops/issues/393
    X = X.contiguous().to(torch.float32)
    C = C.contiguous().to(torch.float32)

    # Use PyKeOps LazyTensors for efficient computation
    x_i = LazyTensor(X[:, :, None, :])  # (B, n, 1, d_m)
    c_j = LazyTensor(C[:, None, :, :])  # (B, 1, c, d_m)

    # Compute squared Euclidean distances
    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, n, c)

    # Get indices of the nearest codewords
    indices = D_ij.argmin(dim=2)  # (B, n)

    # Reshape indices back to (bs, num_heads, n, M)
    indices = indices.view(bs, num_heads, M, n).permute(0, 1, 3, 2)

    return indices.contiguous().to(target_dtype)
    


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

def nbits2dtype(nbits):
    if nbits <= 8:
        return torch.uint8
    elif nbits <= 16:
        return torch.uint16
    elif nbits <= 32:
        return torch.uint32
    elif nbits <= 64:
        return torch.uint64
    else:
        raise ValueError("nbits must be <= 64")


def train_pq_kmeans(X, M, nbits, niter: int = 30) -> torch.Tensor:
    import numpy as np
    import faiss
    n, d = X.shape
    assert d % M == 0, "d must be divisible by M"

    d_sub = d // M  
    n_centroids = 2**nbits 

    X_reshape = X.reshape(-1, d_sub)
    # X_reshape = np.ascontiguousarray(X_reshape)

    kmeans = faiss.Kmeans(d=d_sub, k=n_centroids, niter=niter, verbose=True)
    kmeans.train(X_reshape)

    cent = kmeans.centroids.reshape(1, n_centroids, d_sub)
    cent = torch.from_numpy(cent)
    
    return cent