from argparse import ArgumentParser
import json
import os

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer , AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from datasets import load_dataset
from functools import partial
import tqdm


def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]


def get_k_hook(m, args, kwargs, result, name, output_dict):
    assert isinstance(kwargs, dict)

    hidden_states = kwargs["hidden_states"]
    position_embeddings = kwargs["position_embeddings"]

    bsz, q_len, _ = hidden_states.size()

    q = m.q_proj(hidden_states).view(bsz, q_len, m.config.num_attention_heads, m.head_dim).transpose(1, 2)
    k = m.k_proj(hidden_states).view(bsz, q_len, m.config.num_key_value_heads, m.head_dim).transpose(1, 2)
    v = m.v_proj(hidden_states).view(bsz, q_len, m.config.num_key_value_heads, m.head_dim).transpose(1, 2)

    kv_seq_len = k.shape[-2]
    if isinstance(m, LlamaAttention):
        # print(f"LlamaAttention")
        cos, sin = position_embeddings
        q, k = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
    elif isinstance(m, MistralAttention):
        # print(f"MistralAttention")
        cos, sin = position_embeddings
        q, k = transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb(q, k, cos, sin)
    elif isinstance(m, Qwen2Attention):
        # print(f"Qwen2Attention")
        cos, sin = position_embeddings
        q, k = transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    elif isinstance(m, Qwen3Attention):
        # print(f"Qwen3Attention")
        q = m.q_norm(q)
        k = m.k_norm(k)
        cos, sin = position_embeddings
        q, k = transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    
    out = k.abs().max(dim=-2, keepdim=True).values.pow(0.5).cpu().detach()

    if name+".k_proj" not in output_dict:
        output_dict[name+".k_proj"] = out
    else:
        output_dict[name+".k_proj"] += out

@torch.no_grad()
def get_calib_feat(model: nn.Module, tokenizer):
    output_dict = dict()

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, LlamaAttention) or isinstance(m, MistralAttention) or isinstance(m, Qwen2Attention) or isinstance(m, Qwen3Attention):
            hooks.append(
                m.register_forward_hook(
                    partial(get_k_hook, name=name, output_dict=output_dict), with_kwargs=True))

    print("Collecting activation scales...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    samples = get_calib_dataset(tokenizer, n_samples=256, block_size=512)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return output_dict, len(samples)


if __name__ == '__main__':
    parser = ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_path', type=str, required=True, help='Selected model')
    parser.add_argument('--output_dir', type=str, default="scales/", help='Output directory')
    
    args = parser.parse_args()

    model_path = args.model_path
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    print(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    output_dict, len_samples = get_calib_feat(model, tokenizer)

    channel_config = dict()

    for k, v in output_dict.items():
        channel_config[k] = v / len_samples
    #model_owner, model_name = model_path.split("/")
    path_parts = [p for p in model_path.split("/") if p.strip()]  
    model_owner = path_parts[-2]
    model_name = path_parts[-1]
    
    output_dir = os.path.join(args.output_dir, model_owner)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, model_name + ".pt")

    torch.save(channel_config, output_path)
    print(f"Saved tensor calibration config to: {output_path}")