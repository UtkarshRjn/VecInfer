# H100 measurements, 13 Nov 2026

Llama-3.1-8B-Instruct, d4b8 (sub-vector width 4, 8-bit codes, 2 bits per dimension). Machine: Wolverine H100 80 GB.

Two codebook setups are compared.

- **Separate.** One key codebook and one value codebook per layer. This is the paper's setup.
- **Joint.** One codebook shared by keys and values.

The files in this directory are the structured measurements. `run.log`, `latency.log`, and `fit.log` are the raw logs from the first H100 session. `joint-postsoftmax/` is the later session that trained one codebook on the post-softmax loss and then measured LongBench and latency with that table.

## Joint codebook quality

The joint codebook was trained with k-means on pooled key and value blocks from one Qasper context, truncated to 2048 tokens, 4096 vectors per layer, all 32 layers, 30 iterations. See `joint_kv_qasper.json`.

Block reconstruction error, averaged over layers:

| | Key MSE | Value MSE | Mean MSE |
|---|---:|---:|---:|
| Separate | 0.0713 | 0.0121 | 0.0417 |
| Joint | 0.0775 | 0.0167 | 0.0471 |

The joint mean is **1.130×** the separate mean (range 1.059–1.157 across layers). Keys account for 86% of the separate error and 83% of the joint error.

This is reconstruction error. A LongBench score for the trained joint codebook has not been collected.

## Latency, peak memory, and the 2-bit speedup

Single batch, synthetic prompts, 129 generated tokens, one warmup and five measured repeats. Time to first token is the prefill. Time per output token is the decode. Peak memory is `torch.cuda.max_memory_allocated`, averaged over the five repeats.

The speed test fills the codebooks with random centroids of shape `(1, 256, 4)`. Lookup cost does not depend on the learned centers. Separate uses two independent tables. Joint uses the same table for keys and values. Both are the d4b8 kernel. See `latency.jsonl`.

Decode time per token, milliseconds. The speedup column is FP16 divided by that run.

| Context | FP16 | Separate | Joint | Separate speedup | Joint speedup |
|---:|---:|---:|---:|---:|---:|
| 4k | 20.24 | 24.44 | 20.67 | 0.83× | 0.98× |
| 8k | 25.72 | 24.31 | 18.40 | 1.06× | 1.40× |
| 16k | 36.24 | 23.53 | 19.01 | 1.54× | 1.91× |
| 32k | 57.52 | 21.49 | 19.13 | 2.68× | 3.01× |
| 64k | 102.80 | 25.74 | 21.70 | 3.99× | 4.74× |
| 96k | 157.73 | 25.34 | 24.83 | 6.22× | 6.35× |
| 128k | 205.36 | 28.31 | 27.80 | 7.25× | 7.39× |
| 160k | 253.42 | 31.75 | 31.27 | 7.98× | 8.10× |
| 192k | 301.70 | 34.15 | 33.91 | 8.84× | 8.90× |

At 192k the decode speedup is **8.8×** for separate codebooks and **8.9×** for the joint codebook. The paper reports 8.3× end-to-end decoding speedup for 2-bit VecInfer at this length on an H100. From 96k upward, separate and joint decode times agree within 2%.

Prefill time, milliseconds. 2-bit prefill is slower than FP16 at every length.

| Context | FP16 | Separate | Joint | Separate / FP16 | Joint / FP16 |
|---:|---:|---:|---:|---:|---:|
| 4k | 207 | 305 | 297 | 0.68× | 0.70× |
| 8k | 450 | 641 | 606 | 0.70× | 0.74× |
| 16k | 1129 | 1510 | 1447 | 0.75× | 0.78× |
| 32k | 2843 | 3419 | 3343 | 0.83× | 0.85× |
| 64k | 8312 | 9475 | 9199 | 0.88× | 0.90× |
| 96k | 16827 | 18580 | 18386 | 0.91× | 0.92× |
| 128k | 28572 | 30655 | 30431 | 0.93× | 0.94× |
| 160k | 43215 | 46398 | 46090 | 0.93× | 0.94× |
| 192k | 60617 | 64137 | 63810 | 0.95× | 0.95× |

Peak allocated memory, MiB. Separate and joint match at every length.

| Context | FP16 | 2-bit |
|---:|---:|---:|
| 4k | 16327 | 15895 |
| 8k | 17305 | 16425 |
| 16k | 19261 | 17485 |
| 32k | 23174 | 19606 |
| 64k | 30999 | 23847 |
| 96k | 38824 | 28088 |
| 128k | 46650 | 32329 |
| 160k | 54475 | 36570 |
| 192k | 62302 | 40812 |

At 192k, FP16 peaks at 60.8 GiB and 2-bit peaks at 39.9 GiB. Sharing the codebook does not change the cache size: both setups store 2 bits per dimension.

## Attention-fitted separate codebooks

Separate key and value codebooks were then fit to attention-output error on 1024 Qasper tokens, 20 Adam steps, learning rate 0.01. Mean attention-output MSE fell from 0.003128 to 0.001644, a ratio of **0.537**. Key and value block error rose slightly. Per-layer numbers are in `attention_output_loss.json`. The 64 centroid files are under `centroids/qasper-attn/`.

LongBench used those separate centroids (`cluster_dataset=qasper-attn`), residual length 128, 16 sink tokens. Eight tasks finished. Five stopped on a CUDA illegal-address error inside the KeOps assignment and were not retried. Checkpoints are in `checkpoints/`. Scores are in `longbench_attention_fit.jsonl`.

| Task | Attention-fit | K-means reproduction | Paper d4b8 |
|---|---:|---:|---:|
| Qasper | 45.61 | 45.73 | 46.1 |
| HotpotQA | 54.61 | 54.77 | 55.0 |
| 2WikiMQA | 47.00 | 45.93 | 46.5 |
| TREC | 71.50 | 72.50 | 72.0 |
| TriviaQA | 92.32 | 91.99 | 92.2 |
| PassageCount | 7.55 | 7.83 | 7.9 |
| PassageRetrieval | 99.00 | 99.00 | 98.5 |
| GovReport | 30.07 | 31.07 | 31.2 |
| Mean of these 8 | 55.96 | 56.10 | 56.18 |

The mean is over the eight finished tasks only. It is not the 13-task mean. The paper's 13-task mean is 52.7.

Tasks that did not finish:

| Task | Stopped at | Paper d4b8 |
|---|---|---:|
| MultiFieldQA | example 75 | 53.1 |
| SAMSum | example 36 | 41.6 |
| LCC | example 1 | 60.9 |
| RepoBench-P | example 309 of 500 | 53.2 |
| MultiNews | example 9 | 26.5 |

## Shared codebook trained on the post-softmax loss

One codebook per layer, used for both keys and values. It was initialized by k-means on the pooled key and value blocks, then moved by Adam on the causal attention-output error. Same protocol as the separate fit: 1024 Qasper tokens, 256 centers of width 4, 20 steps. The layer-0 learning-rate search picked 0.001. Per-layer numbers are in `joint-postsoftmax/attention_output_loss.json`. The 32 saved tables are `joint-postsoftmax/centroids/joint_cent_layer*_32_8.pq.pt`. The key and value files loaded at evaluation were copies of that same table.

Both losses, mean over 32 layers. Block error is the mean squared error of the 4-number slices. Attention error is the mean squared error of the causal attention output.

| | Block error, start → after Adam | Attention error, start → after Adam | Mean of the 32 attention ratios |
|---|---:|---:|---:|
| Separate key and value tables | 0.04002 → 0.04238 | 0.003128 → 0.001644 | 0.537 |
| One shared table | 0.04520 → 0.04532 | 0.003108 → 0.002535 | 0.785 |

After Adam, the shared table is 1.07× the separate block error and 1.54× the separate attention error. The block error barely moved. Key block error went from 0.07434 to 0.07461, and value block error from 0.01606 to 0.01602.

LongBench loaded those shared tables (`cluster_dataset=qasper-attn-joint`), residual length 128, 16 sink tokens. Eight tasks finished. The same five tasks stopped on the CUDA illegal-address error, at the same examples as the separate-table run. Checkpoints are in `joint-postsoftmax/checkpoints/`. The score file is `joint-postsoftmax/longbench.jsonl`. The raw log is `joint-postsoftmax/eval.log`.

| Task | Shared post-softmax | Separate post-softmax | Paper d4b8 |
|---|---:|---:|---:|
| Qasper | 43.94 | 45.61 | 46.1 |
| HotpotQA | 55.33 | 54.61 | 55.0 |
| 2WikiMQA | 47.28 | 47.00 | 46.5 |
| TREC | 71.50 | 71.50 | 72.0 |
| TriviaQA | 92.05 | 92.32 | 92.2 |
| PassageCount | 7.57 | 7.55 | 7.9 |
| PassageRetrieval | 99.50 | 99.00 | 98.5 |
| GovReport | 30.08 | 30.07 | 31.2 |
| Mean of these 8 | 55.91 | 55.96 | 56.18 |

The mean is over the eight finished tasks only.

The latency sweep in this session loaded the trained centroids, not random tables. FP16 is the baseline. Separate loads the attention-fitted key and value tables. Joint loads the shared post-softmax table for both. Same lengths, 129 generated tokens, one warmup and five repeats. See `joint-postsoftmax/latency.jsonl`.

Decode time per token, milliseconds.

| Context | FP16 | Separate | Shared | Separate speedup | Shared speedup |
|---:|---:|---:|---:|---:|---:|
| 4k | 20.51 | 18.16 | 23.87 | 1.13× | 0.86× |
| 8k | 25.86 | 18.21 | 23.38 | 1.42× | 1.11× |
| 16k | 36.39 | 19.69 | 23.57 | 1.85× | 1.54× |
| 32k | 57.79 | 19.68 | 24.26 | 2.94× | 2.38× |
| 64k | 103.21 | 22.48 | 24.59 | 4.59× | 4.20× |
| 96k | 158.13 | 25.52 | 26.86 | 6.20× | 5.89× |
| 128k | 205.98 | 28.87 | 30.12 | 7.13× | 6.84× |
| 160k | 253.91 | 31.86 | 32.21 | 7.97× | 7.88× |
| 192k | 301.62 | 35.60 | 34.61 | 8.47× | 8.71× |

At 192k the trained separate tables are 8.47× and the trained shared table is 8.71×. Peak memory matches the random-table sweep: 62302 MiB for FP16 and 40812 MiB for both 2-bit runs. Prefill at 192k is 0.95× FP16 for both (60591 ms FP16, 63949 ms separate, 63651 ms shared).

## What is still open

LongBench for the earlier joint codebook, the one trained by k-means on block error only. The five tasks that hit the CUDA illegal-address error failed again on this retry, at the same examples.
