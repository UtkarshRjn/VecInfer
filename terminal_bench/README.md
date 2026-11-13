# VecInfer on Terminal-Bench lite 2.0

Harbor is the harness for Terminal-Bench. The lite 2.0 set it ships is
[OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite)
(`openthoughts-tblite@2.0`): 100 tasks that track Terminal-Bench 2.0 and are
short enough to iterate on an 8B model. Full Terminal-Bench 2.0 is
`terminal-bench@2.0`.

VecInfer quantizes the KV cache, not the weights. This eval serves
Llama-3.1-8B-Instruct twice, with the same greedy Terminus-2 agent:

| Condition | What changes | Bits per KV element |
| --- | --- | --- |
| `baseline` | stock attention, full-precision cache | 16 (fp16) |
| `d4b8` | VecInfer, paper config | 2 |
| `d8b8`, `d8b12`, `d4b12`, `d2b8` | other codebooks already in `pod-backup/centroids` | 1, 1.5, 3, 4 |

The gap in resolution rate is the effect of KV quantization. Temperature is 0
so the two jobs differ only in the cache.

The quantizer is a single global cache of batch size 1. The Harbor job runs
one trial at a time, and the server resets that cache on every completion.
Restart the server to change bit-width. Do not point two Harbor jobs at one server.

## Setup

Use the existing VecInfer environment (CUDA, the compiled kernels, and
`fast-hadamard-transform`). From `VecInfer/`:

```bash
make bindings
```

`d4b8` uses the 8-bit CUDA kernel. Install Harbor on the machine that will
start the containers (Docker is required):

```bash
pip install harbor
```

Weights, channel scales, and Qasper codebooks are already under `pod-backup/`.

## Run

From the repo root, in the VecInfer environment:

```bash
python -m terminal_bench.serve_vecinfer --preset d4b8 --check-codebooks
python -m terminal_bench.serve_vecinfer --preset d4b8
```

In another shell, smoke-test two tasks, then the full lite set:

```bash
python -m terminal_bench.run_eval --condition d4b8 --n-tasks 2 --launch
python -m terminal_bench.run_eval --condition d4b8 --launch
```

Stop the server and repeat for the baseline:

```bash
python -m terminal_bench.serve_vecinfer --baseline
python -m terminal_bench.run_eval --condition baseline --launch
```

`--launch` checks `/health` and refuses to start if the server preset does not
match the job. Harbor 0.25 runs Terminus-2's model calls on the host, so the
default base URL is `http://127.0.0.1:8000/v1`. Pass
`--base-url http://host.docker.internal:8000/v1` if the agent runs inside the
task container.

Llama 3.1 8B often answers with a markdown script instead of Terminus-2's JSON
command object, and Harbor's default summarizer fires on an 8192-token context
before any command runs. The server rewrites those drafts into JSON commands,
and the job disables proactive summarization.

Full Terminal-Bench 2.0, or the 10-task sample:

```bash
python -m terminal_bench.run_eval --dataset terminal-bench@2.0 --condition d4b8 --launch
python -m terminal_bench.run_eval --dataset sample --condition d4b8 --n-tasks 1 --launch
```

## Read the score

Harbor writes `terminal_bench/jobs/<job_name>/result.json`. Resolution rate is
the mean task reward.

```bash
python -m terminal_bench.run_eval --summarize terminal_bench/jobs/tblite-d4b8-tblite
python -m terminal_bench.run_eval --compare \
  terminal_bench/jobs/tblite-baseline-tblite \
  terminal_bench/jobs/tblite-d4b8-tblite
```

`harbor view terminal_bench/jobs` opens the trajectory viewer.

On an 8B model this lite set is a hard agent eval, not a long-context QA
score. A resolution rate near the floor can still be a valid comparison: use
the baseline job as the control, and treat a drop from baseline to `d4b8` as
the quantization cost. Raise `--max-turns` if trajectories are cut off before
the task timeout. Keep `--max-input-tokens` equal to the server's
`--max-model-len` minus `--max-new-tokens` (defaults are 16384 and 2048).
