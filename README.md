<h2 align="center">VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization</h2>

## Abstract

The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both long-context understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length.

## Getting Started
### Environment Setup

> [!NOTE]
> Environment parameters: CUDA 12.4

Create a Conda Environment:
```
conda create -n vecinfer python=3.12
conda activate vecinfer
``` 
Install Dependencies:
```
pip install -r requirements.txt
cd scripts/utils/evaluation/latex2sympy && pip install -e .
```

Install Fast Hadamard Transform:
```
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform && git checkout v1.0.4
python setup.py install && cd ..
```

### Install Kernels
Compile CUDA Extensions:
```
make bindings
```
## Usage
Calibrate Key Cache Channel Scales:
```
python scripts/utils/calibration.py --model_path /PATH/TO/MODEL --output_dir ./scales
```
Configure Model Parameters:

> Edit a configuration file under `scripts/modeldb/configs/` with the following structure:
```
{
    "model_name" : "llama-3.1-8b-instruct",
    "max_length" : 127500,
    "d" : 128,
    "n_heads" : 32,
    "n_layers" : 32,
    "scales_path" : "/PATH/TO/SCALES",
    "folder" : "/PATH/TO/MODEL"
}
```

### Codebook Generation
We provide the script to generate the codebooks for the KV caches of different models.
```
make training
```
Main arguments explanation:
- `--dataset`: Name of the dataset used to build the clustering codebook.
- `-f`: Path to the model configuration file.
- `-M_key` / `-M_value`: Each sub-vector has a length of `128 / M`.
- `--nbits_key` / `--nbits_value`: Each codebook contains `2 ** nbits` sub-vectors.


> [!TIP] 
> We also release codebooks for models from the `Llama`, `Mistral`, and `Qwen` families on [Huggingface](https://huggingface.co/ydydy/vecinfer-codebooks), which are generated using the Qasper dataset.
> | Base Model | HF Link |
> |---|---|
> | Llama-3.1-8B-Instruct | [Link](https://huggingface.co/ydydy/vecinfer-codebooks/tree/main/llama-3.1-8b-instruct/qasper) |
> | Mistral-7B-Instruct-v0.3 | [Link](https://huggingface.co/ydydy/vecinfer-codebooks/tree/main/mistral-7b-instruct-v0.3/qasper)  |
> | Qwen2.5-14B-Instruct | [Link](https://huggingface.co/ydydy/vecinfer-codebooks/tree/main/qwen2.5-14b-instruct/qasper)  |
> | Qwen3-8B | [Link](http://huggingface.co/ydydy/vecinfer-codebooks/tree/main/qwen3-8b/qasper) |
> | DeepSeek-R1-Distill-Llama-8B | [Link](https://huggingface.co/ydydy/vecinfer-codebooks/tree/main/deepseek-r1-distill-llama-3-8b/qasper) |
> | DeepSeek-R1-Distill-Qwen-14B | [Link](https://huggingface.co/ydydy/vecinfer-codebooks/tree/main/deepseek-r1-distill-qwen-14b/qasper) |
### Accuracy Evaluation
#### LongBench

Download the datasets:
```
python scripts/utils/download.py --dataset longbench 
```
Run LongBench with `VecInfer`:
```
make longbench
```
Main arguments explanation:
- `--dataset`: Name of the dataset used for evaluation.
- `--cluster_dataset`: Name of the dataset used to build the clustering codebook.
- `-f`: Path to the model configuration file.
- `-M_key` / `-M_value`: Each sub-vector has a length of `128 / M`.
- `--nbits_key` / `--nbits_value`: Each codebook contains `2 ** nbits` sub-vectors.

> [!NOTE]
> Results will be saved to `scripts/modeldb/results.jsonl` for possible further analysis.

#### MATH
To run the MATH benchmark, simply run:
```
make math_evaluation
```
- `--dataset`: Name of the dataset used for evaluation.
- `--cluster_dataset`: Name of the dataset used to build the clustering codebook.
- `-f`: Path to the model configuration file.
- `-M_key` / `-M_value`: Each sub-vector has a length of `128 / M`.
- `--nbits_key` / `--nbits_value`: Each codebook contains `2 ** nbits` sub-vectors.
- `--save_path`: Path to the output directory.


To evaluate benchmark results, simply run:
```
cd scripts/utils && bash eval_math.sh
```

> [!NOTE] 
> Results will be saved to `./math_result` for possible further analysis.

### Efficiency Evaluation
End-to-end Efficiency:
```
make e2e
```

Latency Breakdown:
```
make breakdown
```

Kernel-level Efficiency: 
```
cd scripts/modeldb/bindings
bash run_tests.sh
```
> [!NOTE]
> Or utilize NVIDIA Nsight Compute:
> ```
> ncu --target-processes all --set full --export result.ncu-rep python debug.py
> ```

## Citation
```
@misc{yao2025vecinferefficientllminference,
      title={VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization}, 
      author={Dingyu Yao and Chenxu Yang and Zhengyang Tong and Zheng Lin and Wei Liu and Jian Luan and Weiping Wang},
      year={2025},
      eprint={2510.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06175}, 
}
```
