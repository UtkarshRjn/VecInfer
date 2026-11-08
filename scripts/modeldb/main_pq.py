import argparse
import json
import itertools
from tqdm import tqdm
import pathlib
import os
import importlib
import glob
import re

from ..utils.Namespace import UniConfig, load_config
from ..utils.Timer import tprint, Timer
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
    Timer('').start()
    # ================== Argument Parsing ==================
    parser = argparse.ArgumentParser(description="ModelDB")
    parser.add_argument("-f", "--file", type=str, help="Relative path to config.json. Relative to scripts/modeldb/configs/", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=False)
    
    parser.add_argument("-M_key", type=int, help="PQ config, number of sub-sections (key)", required=False)
    parser.add_argument("-M_value", type=int, help="PQ config, number of sub-sections (value)", required=False)
    parser.add_argument("--nbits_key", type=int, help="PQ config, number of bits per sub-section (key)", required=False)
    parser.add_argument("--nbits_value", type=int, help="PQ config, number of bits per sub-section (value)", required=False)
    parser.add_argument("-m", "--merged_training", action="store_true", help="Train a merged PQ", required=False)
    parser.add_argument("--opq", action="store_true", help="Prepand LT before PQ to maximize squred error across dimensions", required=False)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--half", action="store_true", help="Use half precision", required=False)
    parser.add_argument("--breakdown", action="store_true", help="Breakdown timing, could lead to overhead due to additional torch.cuda.synchronize", required=False)
    parser.add_argument(
        "-p", "--pipeline",
        nargs='+',
        choices=["baseline", "sampling", "training", "evaluation"],
        help="List of pipeline stages to execute"
    )
    parser.add_argument("--scales_path", type=str, help="Scales path", required=False)
    parser.add_argument("--save_path", type=str, help="Save path", required=False)
    parser.add_argument("--cluster_dataset", type=str, help="Cluster dataset name", required=False)
    parser.add_argument("--sink_tokens", type=int, help="Sink token", required=False)
    parser.add_argument("--joint_kv", action="store_true", help="Train one codebook on keys and values together and report reconstruction error against separate codebooks", required=False)

    args = parser.parse_args()

    if args.opq:
        raise NotImplementedError("OPQ is not implemented for GPU yet.")
    if args.merged_training is False:
        raise NotImplementedError("Only merged training is supported😈. Use --merged_training")
    
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda' # TODO: support multi-gpu

    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file

    # Load config

    config += load_config(config.config_path)


    if args.M_key is not None:
        config.M_key = args.M_key
    if args.M_value is not None:
        config.M_value = args.M_value
    if args.nbits_key is not None:
        config.nbits_key = args.nbits_key
    if args.nbits_value is not None:
        config.nbits_value = args.nbits_value
    config.joint_kv = bool(args.joint_kv)
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.pipeline is not None:
        config.pipeline = args.pipeline
    if args.seed is not None:
        config.seed = args.seed
    if args.half is not None:
        config.half = args.half
    if args.breakdown is not None:
        config.breakdown = args.breakdown
    if args.scales_path is not None:
        config.scales_path = args.scales_path
    if args.save_path is not None:
        config.save_path = args.save_path
    if args.cluster_dataset is not None:
        config.cluster_dataset = args.cluster_dataset
    if args.sink_tokens is not None:
        config.sink_tokens = args.sink_tokens

    config.scalar_t = torch.float16 if config.half else torch.float32

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    # config.model_path = config.model_root / config.folder
    config.model_path = config.folder
    config.sample_root = config.root / "kv_samples" / config.model_name / config.dataset
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    # config.codebook_root = config.root / "centroids" / config.model_name / config.cluster_dataset

    config.opq = args.opq
    config.merged_training = args.merged_training

    from transformers import AutoConfig
    from .models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    print(config.model_config.model_type)
    config.context = get_context(config.model_config.model_type)

    from ..utils.pq_utils import nbits2dtype
    config.cache_dtype_key = nbits2dtype(config.nbits_key)
    config.cache_dtype_value = nbits2dtype(config.nbits_value)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    if not (len(config.pipeline) == 1 and "training" in config.pipeline):
        tprint(f"Loading model {config.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with config.context.init_context:
            model = AutoModelForCausalLM.from_pretrained(config.model_path,torch_dtype=torch.float16).to(config.device)
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if config.half:
                model = model.half()

    # ================== baseline ==================
    if "baseline" in config.pipeline:
        tprint("Baseline")
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.baseline_context:
            score_baseline = benchmark(model, tokenizer, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score_baseline, "model_name": config.model_name, "dataset": config.dataset, "baseline": True, "half": config.half}))
            f.write("\n")

    # ================== sampling ==================
    if "sampling" in config.pipeline and config.dataset != '_synthetic':
        tprint("Sampling")

        key_pattern = str(config.sample_root / f'key_sampled_layer*_{config.M_key}_{config.nbits_key}.fvecs')
        value_pattern = str(config.sample_root / f'value_sampled_layer*_{config.M_value}_{config.nbits_value}.fvecs')
            
        key_sampled_files = glob.glob(key_pattern)
        value_sampled_files = glob.glob(value_pattern)
        
        if key_sampled_files or value_sampled_files:
            tprint(f"Found {len(key_sampled_files)} key sample files and {len(value_sampled_files)} value sample files")
            tprint(f"Overwrite? (y/n)")
            if input().lower() != 'y':
                tprint("Exit")
                exit()
            else:
                for file_path in key_sampled_files + value_sampled_files:
                    tprint(f"Removing {file_path}")
                    os.remove(file_path)
        
        from .Errors import SamplingComplete
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        config.sampled_nums = 0
        config.expected_sample_nums = 256 * 2**12
        config.expected_vecs_per_sample = config.model_config.num_key_value_heads
        # config.threshold = 1 / config.expected_vecs_per_sample 
        config.threshold = 1/2

        scales = torch.load(config.scales_path, map_location=model.device)
        config.scales = scales

        with config.context.sampling_context:
            try:
                benchmark(model, tokenizer, **(config.to_dict()))
            except SamplingComplete as e:
                tprint(e)
        
        del scales
    # ================== training ==================
    if "training" in config.pipeline and config.dataset != '_synthetic':
        tprint("Training")
        from ..utils.fvecio import read_fvecs
        from ..utils.pq_utils import train_pq_kmeans
        from torch import save

        os.makedirs(config.cent_root, exist_ok=True)
        
        key_pattern = str(config.sample_root / f'key_sampled_layer*_{config.M_key}_{config.nbits_key}.fvecs')

        key_files = sorted(glob.glob(key_pattern))
        
        if not key_files:
            raise FileNotFoundError(f"没有找到采样文件: {key_pattern}")
        
        tprint(f"找到 {len(key_files)} 个 key 采样文件")
        
        value_pattern = str(config.sample_root / f'value_sampled_layer*_{config.M_value}_{config.nbits_value}.fvecs')

        value_files = sorted(glob.glob(value_pattern))
        
        if not value_files:
            raise FileNotFoundError(f"没有找到采样文件: {value_pattern}")
        
        tprint(f"找到 {len(value_files)} 个 value 采样文件")

        if len(key_files) != len(value_files):
            raise ValueError(f"key文件数量({len(key_files)})与value文件数量({len(value_files)})不一致")
        
        
        layer_pattern = re.compile(r'layer(\d+)')
        
        
        for key_file, value_file in zip(key_files, value_files):
            key_match = layer_pattern.search(key_file)
            value_match = layer_pattern.search(value_file)

            if not key_match or not value_match:
                tprint(f"警告: 无法从文件名提取组号: {key_file}, {value_file}")
                continue

            layer_idx = int(key_match.group(1))    

            if key_match.group(1) != value_match.group(1):
                tprint(f"警告: key层号({key_match.group(1)})与value层号({value_match.group(1)})不一致")
                continue
            
                
            tprint(f"正在训练层 {layer_idx}")

            key = read_fvecs(key_file)
            val = read_fvecs(value_file)
            tprint(f"key数据形状: {key.shape}")
            tprint(f"value数据形状: {val.shape}")

            if config.joint_kv:
                if config.M_key != config.M_value or config.nbits_key != config.nbits_value:
                    raise ValueError("joint KV training needs the same M and nbits for keys and values")
                from ..utils.joint_kv import compare_separate_and_joint
                report = compare_separate_and_joint(
                    key, val, config.M_key, config.nbits_key, seed=config.seed
                )
                tprint(
                    f"layer {layer_idx} MSE separate={report['separate_mean_mse']:.6f} "
                    f"joint={report['joint_mean_mse']:.6f} "
                    f"ratio={report['mean_mse_ratio']:.4f}"
                )
                save(torch.from_numpy(report["key_cent"]), config.cent_root / f'key_cent_layer{layer_idx}_{config.M_key}_{config.nbits_key}.pq.pt')
                save(torch.from_numpy(report["value_cent"]), config.cent_root / f'val_cent_layer{layer_idx}_{config.M_value}_{config.nbits_value}.pq.pt')
                save(torch.from_numpy(report["joint_cent"]), config.cent_root / f'joint_cent_layer{layer_idx}_{config.M_key}_{config.nbits_key}.pq.pt')
            else:
                key_cent = train_pq_kmeans(key, config.M_key, config.nbits_key)
                save(key_cent, config.cent_root / f'key_cent_layer{layer_idx}_{config.M_key}_{config.nbits_key}.pq.pt')
                del key_cent

                val_cent = train_pq_kmeans(val, config.M_value, config.nbits_value)
                save(val_cent, config.cent_root / f'val_cent_layer{layer_idx}_{config.M_value}_{config.nbits_value}.pq.pt')
                del val_cent
            del key
            del val
            
        tprint("所有层训练完成")

    if "evaluation" not in config.pipeline:
        tprint("Exit")
        exit()

    # ================== PQ Config ==================
    if "evaluation" in config.pipeline:
        if config.dataset == '_synthetic':
            tprint("Using synthetic centroids for speed evaluation")
            layer_count = config.model_config.num_hidden_layers

            key_cents = []
            val_cents = []
            
            for layer_idx in range(layer_count):
                key_cent = torch.randn(1, 2**config.nbits_key, config.d // config.M_key, dtype=model.dtype, device=config.device)
                val_cent = torch.randn(1, 2**config.nbits_value, config.d // config.M_value, dtype=model.dtype, device=config.device)
                key_cents.append(key_cent)
                val_cents.append(val_cent)
            tprint(f"生成了{len(key_cents)}层的synthetic centroids")
        else:
            config.codebook_root = config.root / "centroids" / config.model_name / config.cluster_dataset
            layer_count = config.model_config.num_hidden_layers
            key_cents = []
            val_cents = []    

            
            for layer_idx in range(layer_count):
                key_cent_path = config.codebook_root / f'key_cent_layer{layer_idx}_{config.M_key}_{config.nbits_key}.pq.pt'
                val_cent_path = config.codebook_root / f'val_cent_layer{layer_idx}_{config.M_value}_{config.nbits_value}.pq.pt'
                tprint(f"key_cent_path: {key_cent_path}")
                tprint(f"val_cent_path: {val_cent_path}")
                
                if not os.path.exists(key_cent_path) or not os.path.exists(val_cent_path):
                    tprint(f"警告: 找不到第{layer_idx}层的参数文件")

                
                key_cent = torch.load(key_cent_path, weights_only=True)
                key_cent = key_cent.to(config.device).to(model.dtype)
                key_cents.append(key_cent)
                
                val_cent = torch.load(val_cent_path, weights_only=True)
                val_cent = val_cent.to(config.device).to(model.dtype)
                val_cents.append(val_cent)                
            
            
            tprint(f"成功加载了{len(key_cents)}层的参数")

        from ..utils.pq_utils import DynamicPQCache
        cache = DynamicPQCache(
            bs = 1, # TODO: support batch size?
            num_key_value_heads=config.model_config.num_key_value_heads,
            nh = config.model_config.num_attention_heads,
            M_key = config.M_key,
            M_value = config.M_value,
            layer_num=config.model_config.num_hidden_layers,
            dtype_key=config.cache_dtype_key,
            dtype_value=config.cache_dtype_value,
            nbits_key=config.nbits_key,
            nbits_value=config.nbits_value,
            d=config.d,
            scalar_t=config.scalar_t,
            scales_path=config.scales_path,
            sink_tokens=config.sink_tokens
        )
        cache.set_cent(key_cents, val_cents)
    
        tprint("Evaluation")
 
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]
        torch.cuda.empty_cache()
        with config.context.evaluation_context:
            score = benchmark(model, tokenizer, cache_clear_func=cache.init_cache, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score, **config.to_serializable_dict()}))
            f.write("\n")

    tprint("Exit")
