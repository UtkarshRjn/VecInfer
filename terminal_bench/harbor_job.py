"""Harbor job config and result summary for a VecInfer Terminal-Bench run.

The lite Terminal-Bench 2.0 set registered by Harbor is OpenThoughts-TBLite
(``openthoughts-tblite@2.0``): 100 tasks calibrated to track Terminal-Bench 2.0.
``terminal-bench@2.0`` is the full 89-task benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS: dict[str, tuple[str, str]] = {
    "tblite": ("openthoughts-tblite", "2.0"),
    "lite": ("openthoughts-tblite", "2.0"),
    "openthoughts-tblite": ("openthoughts-tblite", "2.0"),
    "openthoughts-tblite@2.0": ("openthoughts-tblite", "2.0"),
    "terminal-bench": ("terminal-bench", "2.0"),
    "terminal-bench@2.0": ("terminal-bench", "2.0"),
    "tb2": ("terminal-bench", "2.0"),
    "terminal-bench-sample": ("terminal-bench-sample", "2.0"),
    "terminal-bench-sample@2.0": ("terminal-bench-sample", "2.0"),
    "sample": ("terminal-bench-sample", "2.0"),
}

SERVED_MODEL = "vecinfer-llama-3.1-8b-instruct"


def resolve_dataset(name: str) -> tuple[str, str]:
    key = name.strip()
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Use one of: {known}")
    return DATASETS[key]


def build_job_config(
    *,
    job_name: str,
    jobs_dir: str,
    dataset: str,
    base_url: str,
    max_input_tokens: int = 16384,
    max_output_tokens: int = 2048,
    max_turns: int = 40,
    temperature: float = 0.0,
    n_tasks: int | None = None,
    task_names: list[str] | None = None,
    n_attempts: int = 1,
    agent_timeout_multiplier: float | None = None,
) -> dict[str, Any]:
    dataset_name, version = resolve_dataset(dataset)
    dataset_entry: dict[str, Any] = {"name": dataset_name, "version": version}
    if n_tasks is not None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be at least 1")
        dataset_entry["n_tasks"] = n_tasks
    if task_names:
        dataset_entry["task_names"] = list(task_names)

    # One trial at a time. VecInfer's KV cache is a batch-size-1 singleton.
    config: dict[str, Any] = {
        "job_name": job_name,
        "jobs_dir": jobs_dir,
        "n_attempts": n_attempts,
        "n_concurrent_trials": 1,
        "quiet": False,
        "metrics": [{"type": "mean"}],
        "environment": {
            "type": "docker",
            "extra_allowed_hosts": ["host.docker.internal"],
        },
        "agents": [
            {
                "name": "terminus-2",
                "model_name": f"openai/{SERVED_MODEL}",
                "base_url": base_url.rstrip("/"),
                "model_api_key_name": "OPENAI_API_KEY",
                "model_api_format": "openai_chat_completions",
                "n_concurrent": 1,
                "extra_allowed_hosts": ["host.docker.internal"],
                "env": {"OPENAI_API_KEY": "vecinfer"},
                "kwargs": {
                    "temperature": temperature,
                    "max_turns": max_turns,
                    "parser_name": "json",
                    "enable_summarize": True,
                    "suppress_max_turns_warning": True,
                    "model_info": {
                        "max_input_tokens": max_input_tokens,
                        "max_output_tokens": max_output_tokens,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            }
        ],
        "datasets": [dataset_entry],
    }
    if agent_timeout_multiplier is not None:
        config["agent_timeout_multiplier"] = agent_timeout_multiplier
    return config


def write_job_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def _reward_mean(reward_stats: dict[str, Any]) -> tuple[float | None, int]:
    """Mean of the ``reward`` key. Values map to trial-name lists."""
    if not reward_stats:
        return None, 0
    primary = reward_stats.get("reward")
    if not isinstance(primary, dict):
        primary = next((v for v in reward_stats.values() if isinstance(v, dict)), None)
    if not isinstance(primary, dict) or not primary:
        return None, 0
    total = 0
    weighted = 0.0
    for raw_value, trials in primary.items():
        if not isinstance(trials, list):
            continue
        count = len(trials)
        total += count
        weighted += float(raw_value) * count
    if total == 0:
        return None, 0
    return weighted / total, total


def _trials_from_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in payload.get("trial_results") or []:
        if not isinstance(trial, dict):
            continue
        rewards = None
        verifier = trial.get("verifier_result") or {}
        if isinstance(verifier, dict):
            rewards = verifier.get("rewards")
        reward = None
        if isinstance(rewards, dict) and "reward" in rewards:
            reward = float(rewards["reward"])
        elif isinstance(rewards, dict) and rewards:
            reward = float(next(iter(rewards.values())))
        name = str(trial.get("trial_name") or trial.get("task_name") or "")
        task = name.split("__", 1)[0]
        rows.append(
            {
                "task": task,
                "trial_name": name,
                "reward": reward,
                "reward_valid": bool(trial.get("reward_valid", reward is not None)),
            }
        )
    return rows


def summarize_job(path: Path) -> dict[str, Any]:
    """Summarize one Harbor ``result.json`` or a job directory containing it."""
    result_path = path
    if path.is_dir():
        direct = path / "result.json"
        if direct.is_file():
            result_path = direct
        else:
            found = sorted(path.glob("**/result.json"))
            if not found:
                raise FileNotFoundError(f"No result.json under {path}")
            result_path = found[-1]
    payload = json.loads(result_path.read_text())
    stats = payload.get("stats") or {}
    evals = stats.get("evals") or {}
    eval_rows = []
    for key, ev in evals.items():
        if not isinstance(ev, dict):
            continue
        mean, n_rewards = _reward_mean(ev.get("reward_stats") or {})
        eval_rows.append(
            {
                "eval": key,
                "n_trials": ev.get("n_trials"),
                "n_errors": ev.get("n_errors"),
                "resolution_rate": mean,
                "n_scored": n_rewards,
                "metrics": ev.get("metrics") or [],
            }
        )
    trials = _trials_from_result(payload)
    scored = [row["reward"] for row in trials if isinstance(row["reward"], float)]
    trial_mean = sum(scored) / len(scored) if scored else None
    if eval_rows and eval_rows[0]["resolution_rate"] is not None:
        resolution = eval_rows[0]["resolution_rate"]
        n_scored = eval_rows[0]["n_scored"]
    else:
        resolution = trial_mean
        n_scored = len(scored)
    return {
        "result_path": str(result_path),
        "n_total_trials": payload.get("n_total_trials"),
        "n_completed_trials": stats.get("n_completed_trials"),
        "n_errored_trials": stats.get("n_errored_trials"),
        "resolution_rate": resolution,
        "n_scored": n_scored,
        "evals": eval_rows,
        "trials": trials,
    }


def format_summary(summary: dict[str, Any]) -> str:
    rate = summary["resolution_rate"]
    rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
    lines = [
        f"result: {summary['result_path']}",
        f"resolution rate: {rate_text} ({summary['n_scored']} scored)",
        (
            f"completed: {summary['n_completed_trials']}  "
            f"errored: {summary['n_errored_trials']}  "
            f"total: {summary['n_total_trials']}"
        ),
    ]
    for row in summary["trials"]:
        reward = "n/a" if row["reward"] is None else f"{row['reward']:.0f}"
        lines.append(f"  {row['task']}: {reward}")
    return "\n".join(lines)


def compare_summaries(left: dict[str, Any], right: dict[str, Any]) -> str:
    def rate(summary: dict[str, Any]) -> str:
        value = summary["resolution_rate"]
        if value is None:
            return "n/a"
        return f"{100 * value:.2f}% ({summary['n_scored']})"

    lines = [
        f"A {rate(left)}  {left['result_path']}",
        f"B {rate(right)}  {right['result_path']}",
    ]
    by_task_b = {row["task"]: row["reward"] for row in right["trials"]}
    for row in left["trials"]:
        other = by_task_b.get(row["task"])
        if row["reward"] is None and other is None:
            continue
        lines.append(
            f"  {row['task']}: A={row['reward']} B={other}"
        )
    return "\n".join(lines)
