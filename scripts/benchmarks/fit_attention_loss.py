"""Fit per-layer KV codebooks to causal attention-output error on one Qasper context."""

import inspect
import json
import math
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from attention_codebook import fit_attention_codebooks


def hadamard_transform(x, scale):
    d = x.shape[-1]
    y = x.float().contiguous()
    h = 1
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], d)
        h *= 2
    return y * scale


def repeat_kv(x, groups):
    batch, heads, tokens, dim = x.shape
    x = x.unsqueeze(2).expand(batch, heads, groups, tokens, dim)
    return x.reshape(batch, heads * groups, tokens, dim)


def choose_lr(q, k, v):
    best_lr = None
    best_ratio = None
    for lr in (1e-3, 1e-2, 1e-1):
        report = fit_attention_codebooks(q, k, v, steps=20, lr=lr, kmeans_iters=15)
        log(f"lr {lr:.0e} output {report['kmeans_output_mse']:.6e} -> {report['fitted_output_mse']:.6e} ratio {report['output_mse_ratio']:.4f}")
        if best_ratio is None or report["output_mse_ratio"] < best_ratio:
            best_lr = lr
            best_ratio = report["output_mse_ratio"]
    return best_lr


def log(message):
    print(message, flush=True)
    with open("/workspace/fit.log", "a") as handle:
        handle.write(message + "\n")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    model_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    log(f"threads={torch.get_num_threads()}")
    scales = torch.load("/workspace/scales.pt", map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    model.to(device).eval()

    dataset = Dataset.from_file("/workspace/qasper.arrow")
    order = sorted(range(len(dataset)), key=lambda i: int(dataset[i]["length"]), reverse=True)
    text = dataset[order[0]]["context"]
    max_length = int(os.environ.get("ATTN_SEQ", "1024"))
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    log(f"tokens={input_ids.shape[1]}")

    captured = {}

    def make_forward(attn, original):
        signature = inspect.signature(original)

        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            hidden = bound.arguments["hidden_states"]
            cos, sin = bound.arguments["position_embeddings"]
            batch, tokens, _ = hidden.shape
            n_kv = attn.config.num_key_value_heads
            head_dim = attn.head_dim
            query = attn.q_proj(hidden).view(batch, tokens, -1, head_dim).transpose(1, 2)
            key = attn.k_proj(hidden).view(batch, tokens, n_kv, head_dim).transpose(1, 2)
            value = attn.v_proj(hidden).view(batch, tokens, n_kv, head_dim).transpose(1, 2)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            scale = scales[f"model.layers.{attn.layer_idx}.self_attn.k_proj"].to(device=key.device, dtype=key.dtype)
            groups = attn.config.num_attention_heads // n_kv
            query = query * repeat_kv(scale, groups)
            key = key / scale
            hadamard_scale = 1.0 / math.sqrt(key.shape[-1])
            query = hadamard_transform(query, hadamard_scale).to(torch.float16)
            key = hadamard_transform(key, hadamard_scale).to(torch.float16)
            captured[attn.layer_idx] = (
                query[0].detach().float().cpu(),
                key[0].detach().float().cpu(),
                value[0].detach().float().cpu(),
            )
            log(f"captured layer {attn.layer_idx}")
            return original(*args, **kwargs)

        return wrapped

    for layer in model.model.layers:
        attn = layer.self_attn
        attn.forward = make_forward(attn, attn.forward)

    with torch.no_grad():
        model(input_ids=input_ids.to(device), use_cache=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q0, k0, v0 = captured[0]
    lr = choose_lr(q0.to(device), k0.to(device), v0.to(device))
    log(f"selected_lr={lr}")

    rows = []
    for layer_idx in range(len(captured)):
        query, key, value = captured[layer_idx]
        report = fit_attention_codebooks(
            query.to(device), key.to(device), value.to(device),
            steps=int(os.environ.get("ATTN_STEPS", "20")), lr=lr, kmeans_iters=15,
        )
        report["layer"] = layer_idx
        rows.append(report)
        log(
            f"layer {layer_idx:02d} output {report['kmeans_output_mse']:.6e} -> {report['fitted_output_mse']:.6e} "
            f"ratio {report['output_mse_ratio']:.4f} "
            f"key_mse {report['kmeans_key_mse']:.6f}->{report['fitted_key_mse']:.6f} "
            f"value_mse {report['kmeans_value_mse']:.6f}->{report['fitted_value_mse']:.6f}"
        )

    ratios = [row["output_mse_ratio"] for row in rows]
    summary = {
        "model": model_id,
        "tokens": int(input_ids.shape[1]),
        "block_size": 4,
        "nbits": 8,
        "centroids": 256,
        "steps": int(os.environ.get("ATTN_STEPS", "20")),
        "lr": lr,
        "mean_kmeans_output_mse": sum(row["kmeans_output_mse"] for row in rows) / len(rows),
        "mean_fitted_output_mse": sum(row["fitted_output_mse"] for row in rows) / len(rows),
        "mean_output_ratio": sum(ratios) / len(ratios),
        "min_output_ratio": min(ratios),
        "max_output_ratio": max(ratios),
        "layers": rows,
    }
    Path("/workspace/attention_output_loss.json").write_text(json.dumps(summary, indent=2))
    log(
        f"SUMMARY output {summary['mean_kmeans_output_mse']:.6e} -> {summary['mean_fitted_output_mse']:.6e} "
        f"ratio {summary['mean_output_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
