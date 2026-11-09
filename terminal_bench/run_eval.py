"""Write a Harbor job for Terminal-Bench lite 2.0 and optionally launch it.

The served model must already be running. VecInfer can hold one KV-cache
configuration per process, so the full-precision baseline and each bit-width
are separate server processes and separate Harbor jobs.

Example, 2-bit VecInfer (paper config d4b8) on the lite set, two tasks:

    python -m terminal_bench.serve_vecinfer --preset d4b8
    python -m terminal_bench.run_eval --condition d4b8 --n-tasks 2 --launch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from terminal_bench.harbor_job import (
    build_job_config,
    compare_summaries,
    format_summary,
    summarize_job,
    write_job_config,
)
from terminal_bench.quant import PRESETS, bits_per_dimension, preset_to_pq

REPO_ROOT = Path(__file__).resolve().parents[1]


def _job_name(condition: str, dataset: str) -> str:
    safe_dataset = dataset.replace("@", "-").replace("/", "-")
    return f"tblite-{condition}-{safe_dataset}"


def _health(server_url: str) -> dict:
    url = server_url.rstrip("/") + "/health"
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VecInfer on Terminal-Bench lite 2.0")
    parser.add_argument(
        "--dataset",
        default="tblite",
        help="tblite (OpenThoughts-TBLite 2.0), terminal-bench@2.0, or sample",
    )
    parser.add_argument(
        "--condition",
        default="d4b8",
        help="baseline or a VecInfer preset: " + ", ".join(sorted(PRESETS)),
    )
    parser.add_argument("--n-tasks", type=int, default=None, help="Cap the task count for a smoke run")
    parser.add_argument("--task", action="append", default=None, help="Task name glob. Repeatable.")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Scale each task's agent timeout. Local PQ decode is slower than a hosted API.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="URL the Harbor agent uses to reach the VecInfer server. Terminus-2 calls the model from the host, so this is the host address (a kubectl port-forward). Use http://host.docker.internal:8000/v1 only if the agent runs inside the task container.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="URL this process uses for /health",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("terminal_bench/jobs"),
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Where to write the Harbor config. Default: terminal_bench/configs/<job>.json",
    )
    parser.add_argument("--launch", action="store_true", help="Run harbor after writing the config")
    parser.add_argument("--summarize", type=Path, default=None, help="Summarize a Harbor result.json or job dir")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summarize is not None:
        print(format_summary(summarize_job(args.summarize)))
        return
    if args.compare is not None:
        left = summarize_job(args.compare[0])
        right = summarize_job(args.compare[1])
        print(compare_summaries(left, right))
        return

    condition = args.condition
    if condition != "baseline":
        m_sub, nbits = preset_to_pq(condition)
        bits = bits_per_dimension(m_sub, nbits)
        quant_label = f"{condition} ({bits:g} bit/element, M={m_sub}, nbits={nbits})"
    else:
        quant_label = "baseline (full-precision KV cache)"

    job_name = _job_name(condition, args.dataset)
    config = build_job_config(
        job_name=job_name,
        jobs_dir=str(args.jobs_dir),
        dataset=args.dataset,
        base_url=args.base_url,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_turns=args.max_turns,
        temperature=args.temperature,
        n_tasks=args.n_tasks,
        task_names=args.task,
        n_attempts=args.n_attempts,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
    )
    config_path = args.config_out or (REPO_ROOT / "terminal_bench" / "configs" / f"{job_name}.json")
    write_job_config(config_path, config)
    print(f"condition: {quant_label}")
    print(f"config: {config_path}")
    print(f"harbor run --config {config_path}")

    if not args.launch:
        print("Config only. Re-run with --launch once the VecInfer server is up.")
        return

    try:
        health = _health(args.server_url)
    except Exception as exc:
        raise SystemExit(
            f"VecInfer server is not reachable at {args.server_url}/health ({exc}). "
            "Start it with python -m terminal_bench.serve_vecinfer before --launch."
        ) from exc
    server_mode = health.get("mode")
    if condition == "baseline" and server_mode != "baseline":
        raise SystemExit(
            f"This job is the full-precision baseline, but the server is mode={server_mode}. "
            "Restart with python -m terminal_bench.serve_vecinfer --baseline."
        )
    if condition != "baseline":
        if server_mode != "vecinfer":
            raise SystemExit(f"Server mode is {server_mode}, expected vecinfer for {condition}.")
        if health.get("preset") not in (None, condition) and health.get("preset") != condition:
            raise SystemExit(
                f"Server preset is {health.get('preset')}, job condition is {condition}. "
                "Restart the server with that preset."
            )
        expected_m, expected_bits = preset_to_pq(condition)
        if health.get("M") != expected_m or health.get("nbits") != expected_bits:
            raise SystemExit(
                f"Server is M={health.get('M')} nbits={health.get('nbits')}, "
                f"but {condition} is M={expected_m} nbits={expected_bits}."
            )
    if health.get("max_model_len") not in (None, args.max_input_tokens + args.max_output_tokens):
        print(
            "warning: server max_model_len="
            f"{health.get('max_model_len')} but the agent context is "
            f"{args.max_input_tokens} input + {args.max_output_tokens} output. "
            "Pass matching --max-model-len / --max-new-tokens to the server.",
            file=sys.stderr,
        )

    harbor = shutil.which("harbor")
    if harbor is None:
        raise SystemExit("harbor is not on PATH. Install it with: pip install harbor")
    subprocess.run([harbor, "run", "--config", str(config_path)], check=True)
    job_dir = args.jobs_dir / job_name
    if (job_dir / "result.json").is_file() or any(job_dir.glob("result.json")):
        print(format_summary(summarize_job(job_dir)))
    else:
        print(f"Harbor finished. Summarize with: python -m terminal_bench.run_eval --summarize {args.jobs_dir}")


if __name__ == "__main__":
    main()
