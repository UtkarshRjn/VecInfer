"""OpenAI-compatible server for Llama-3.1-8B with VecInfer KV quantization.

Harbor's Terminus-2 agent calls this process. In Harbor 0.25 those calls come
from the host, so publish the server on the host (a port-forward is enough). Each
request is an independent generate() call. VecInfer's product quantizer is a
process-wide batch-size-1 cache, so requests run one at a time and the cache
is reset before every request.

Start this inside the VecInfer environment, on the machine that has the GPU:

    python -m terminal_bench.serve_vecinfer --preset d4b8

Then point Harbor at http://host.docker.internal:8000/v1. See run_eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from terminal_bench.quant import (
    HEAD_DIM,
    N_LAYERS,
    bits_per_dimension,
    missing_codebooks,
    preset_to_pq,
)

def _locate_roots() -> tuple[Path, Path]:
    """Support terminal_bench as a sibling of VecInfer or inside the VecInfer repo."""
    parent = Path(__file__).resolve().parents[1]
    if (parent / "scripts" / "modeldb").is_dir():
        return parent, parent
    nested = parent / "VecInfer"
    if (nested / "scripts" / "modeldb").is_dir():
        return parent, nested
    return parent, parent


REPO_ROOT, VECINFER_ROOT = _locate_roots()
DEFAULT_MODEL = next(
    (
        path
        for path in (
            REPO_ROOT / "pod-backup/models/NousResearch/Meta-Llama-3.1-8B-Instruct",
            REPO_ROOT / "models/Meta-Llama-3.1-8B-Instruct",
            VECINFER_ROOT / "models/Meta-Llama-3.1-8B-Instruct",
        )
        if path.exists()
    ),
    REPO_ROOT / "models/Meta-Llama-3.1-8B-Instruct",
)
DEFAULT_SCALES = next(
    (
        path
        for path in (
            REPO_ROOT / "pod-backup/scales/NousResearch/Meta-Llama-3.1-8B-Instruct.pt",
            REPO_ROOT / "scales/NousResearch/Meta-Llama-3.1-8B-Instruct.pt",
            VECINFER_ROOT / "scales/NousResearch/Meta-Llama-3.1-8B-Instruct.pt",
        )
        if path.exists()
    ),
    REPO_ROOT / "scales/NousResearch/Meta-Llama-3.1-8B-Instruct.pt",
)
DEFAULT_CODEBOOKS = next(
    (
        path
        for path in (
            REPO_ROOT / "pod-backup/centroids/llama-3.1-8b-instruct/qasper",
            REPO_ROOT / "centroids/llama-3.1-8b-instruct/qasper",
            VECINFER_ROOT / "centroids/llama-3.1-8b-instruct/qasper",
        )
        if path.exists()
    ),
    REPO_ROOT / "centroids/llama-3.1-8b-instruct/qasper",
)
SERVED_MODEL = "vecinfer-llama-3.1-8b-instruct"


def _prepare_import_path() -> None:
    root = str(VECINFER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class VecInferEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.mode = "baseline" if args.baseline else "vecinfer"
        self.m_subquantizers = args.m_subquantizers
        self.nbits = args.nbits
        self.bits = (
            None
            if args.baseline
            else bits_per_dimension(args.m_subquantizers, args.nbits, args.head_dim)
        )
        self.model = None
        self.tokenizer = None
        self.cache = None
        self._load()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": self.mode,
            "preset": self.args.preset,
            "M": self.m_subquantizers,
            "nbits": self.nbits,
            "bits_per_dimension": self.bits,
            "sink_tokens": self.args.sink_tokens,
            "max_model_len": self.args.max_model_len,
            "model": SERVED_MODEL,
        }

    def _load(self) -> None:
        # UniConfig nbits are read when modeling_*.py builds model_context.
        # Set them before that import.
        _prepare_import_path()
        import torch
        from scripts.utils.Namespace import UniConfig

        config = UniConfig()
        config.nbits_key = self.nbits
        config.nbits_value = self.nbits
        config.M_key = self.m_subquantizers
        config.M_value = self.m_subquantizers
        config.scales_path = str(self.args.scales)
        config.half = True
        config.sink_tokens = self.args.sink_tokens
        config.distort_recent = False
        config.breakdown = False
        config.device = "cuda"
        config.scalar_t = torch.float16
        config.d = self.args.head_dim

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from scripts.modeldb.models.ModelContext import get_context
        from scripts.utils.pq_utils import nbits2dtype

        model_config = AutoConfig.from_pretrained(self.args.model)
        config.context = get_context(model_config.model_type)
        if model_config.model_type != "llama":
            raise SystemExit(
                f"This server patches Llama attention. Got model_type={model_config.model_type}."
            )

        print(f"Loading {self.args.model} ({self.mode})", flush=True)
        # Stream weights onto the GPU. A Nautilus notebook pod is capped at 32Gi RAM,
        # and this one may be tighter, so avoid a second full copy on the host.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": "cuda"},
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.mode == "baseline":
            self._context = config.context.baseline_context
            self._context.__enter__()
            print("Baseline attention (full-precision KV cache)", flush=True)
            return

        missing = missing_codebooks(
            self.args.codebooks, self.m_subquantizers, self.nbits, model_config.num_hidden_layers
        )
        if missing:
            preview = "\n".join(str(path) for path in missing[:8])
            raise SystemExit(
                f"Missing {len(missing)} codebook files under {self.args.codebooks}:\n{preview}"
            )
        if not Path(self.args.scales).is_file():
            raise SystemExit(f"Scales file not found: {self.args.scales}")

        from scripts.utils.pq_utils import DynamicPQCache

        key_cents = []
        val_cents = []
        for layer in range(model_config.num_hidden_layers):
            key_path = (
                self.args.codebooks
                / f"key_cent_layer{layer}_{self.m_subquantizers}_{self.nbits}.pq.pt"
            )
            val_path = (
                self.args.codebooks
                / f"val_cent_layer{layer}_{self.m_subquantizers}_{self.nbits}.pq.pt"
            )
            key_cents.append(torch.load(key_path, weights_only=True).to("cuda").to(self.model.dtype))
            val_cents.append(torch.load(val_path, weights_only=True).to("cuda").to(self.model.dtype))

        self.cache = DynamicPQCache(
            bs=1,
            num_key_value_heads=model_config.num_key_value_heads,
            nh=model_config.num_attention_heads,
            M_key=self.m_subquantizers,
            M_value=self.m_subquantizers,
            layer_num=model_config.num_hidden_layers,
            dtype_key=nbits2dtype(self.nbits),
            dtype_value=nbits2dtype(self.nbits),
            nbits_key=self.nbits,
            nbits_value=self.nbits,
            d=self.args.head_dim,
            scalar_t=torch.float16,
            scales_path=str(self.args.scales),
            sink_tokens=self.args.sink_tokens,
        )
        self.cache.set_cent(key_cents, val_cents)
        self._context = config.context.evaluation_context
        self._context.__enter__()
        print(
            f"VecInfer KV cache: M={self.m_subquantizers} nbits={self.nbits} "
            f"({self.bits:g} bit/element) sink={self.args.sink_tokens}",
            flush=True,
        )

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.args.max_new_tokens)
        max_new = max(1, min(max_new, self.args.max_new_tokens))
        temperature = body.get("temperature", 0.0)
        temperature = 0.0 if temperature is None else float(temperature)
        do_sample = temperature > 0
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        max_prompt = self.args.max_model_len - max_new
        if max_prompt < 8:
            raise ValueError("max_model_len is smaller than max_new_tokens")

        with self.lock:
            started = time.perf_counter()
            import torch

            input_ids = self._render(messages, max_prompt)
            prompt_tokens = int(input_ids.shape[-1])
            if self.cache is not None:
                self.cache.init_cache()
            stopping = _StopOnStrings(self.tokenizer, prompt_tokens, [str(s) for s in stops if s])
            generate_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": max_new,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "stopping_criteria": stopping.criteria(),
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                if body.get("top_p") is not None:
                    generate_kwargs["top_p"] = float(body["top_p"])
            with torch.inference_mode():
                output = self.model.generate(**generate_kwargs)
            new_tokens = output[0, prompt_tokens:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            for stop in stops:
                index = text.find(str(stop))
                if index >= 0:
                    text = text[:index]
                    break
            elapsed = time.perf_counter() - started
            completion_tokens = int(new_tokens.shape[-1])
            print(
                f"{self.mode} prompt={prompt_tokens} completion={completion_tokens} "
                f"{elapsed:.1f}s",
                flush=True,
            )
            return {
                "id": f"chatcmpl-vecinfer-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or SERVED_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

    def _render(self, messages: list[dict[str, Any]], max_prompt: int):
        import torch

        kept: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role") or "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in (None, "text"):
                        parts.append(str(item.get("text") or ""))
                    elif isinstance(item, str):
                        parts.append(item)
                content = "".join(parts)
            if content is None:
                continue
            kept.append({"role": role, "content": str(content)})
        if not kept:
            raise ValueError("messages contained no text")

        while len(kept) > 1:
            ids = self._tokenize(kept)
            if int(ids.shape[-1]) <= max_prompt:
                return ids.to("cuda")
            drop_at = 1 if kept[0]["role"] == "system" else 0
            if drop_at >= len(kept) - 1:
                break
            kept.pop(drop_at)

        ids = self._tokenize(kept)
        if int(ids.shape[-1]) > max_prompt:
            ids = ids[:, -max_prompt:]
        return ids.to("cuda")

    def _tokenize(self, messages: list[dict[str, str]]):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


class _StopOnStrings:
    def __init__(self, tokenizer, prompt_len: int, stops: list[str]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stops = [stop for stop in stops if stop]
        self.criteria = self._build

    def _build(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        outer = self

        class _Criteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                if not outer.stops:
                    return False
                tail = input_ids[0, outer.prompt_len :]
                if tail.numel() == 0:
                    return False
                text = outer.tokenizer.decode(tail, skip_special_tokens=True)
                return any(stop in text for stop in outer.stops)

        return StoppingCriteriaList([_Criteria()])


ENGINE: VecInferEngine | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/health", "/v1/health"}:
            self._send(200, ENGINE.health() if ENGINE else {"status": "loading"})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SERVED_MODEL,
                            "object": "model",
                            "owned_by": "vecinfer",
                        }
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        if ENGINE is None:
            self._send(503, {"error": {"message": "model is not loaded", "type": "unavailable"}})
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
            if body.get("stream"):
                self._send(
                    400,
                    {
                        "error": {
                            "message": "stream=true is not supported; Terminus uses non-streaming completions",
                            "type": "invalid_request",
                        }
                    },
                )
                return
            self._send(200, ENGINE.chat(body))
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": {"message": str(exc), "type": "server_error"}})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve VecInfer as an OpenAI chat model")
    parser.add_argument("--preset", default="d4b8", help="Paper-style name, e.g. d4b8 (2-bit)")
    parser.add_argument("--baseline", action="store_true", help="Full-precision KV cache")
    parser.add_argument("--m", type=int, default=None, help="Sub-quantizers. Overrides --preset")
    parser.add_argument("--nbits", type=int, default=None, help="Bits per codebook. Overrides --preset")
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scales", type=Path, default=DEFAULT_SCALES)
    parser.add_argument("--codebooks", type=Path, default=DEFAULT_CODEBOOKS)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=18432,
        help="Prompt plus completion cap. Default matches run_eval's 16384 input + 2048 output.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--check-codebooks",
        action="store_true",
        help="Exit after verifying codebook files. Does not load the model.",
    )
    args = parser.parse_args(argv)
    m_sub, nbits = preset_to_pq(args.preset)
    if args.m is not None:
        m_sub = args.m
    if args.nbits is not None:
        nbits = args.nbits
    args.m_subquantizers = m_sub
    args.nbits = nbits
    if not args.baseline:
        bits_per_dimension(m_sub, nbits, args.head_dim)
    return args


def main(argv: list[str] | None = None) -> None:
    global ENGINE
    args = parse_args(argv)
    if args.check_codebooks:
        if args.baseline:
            print("baseline does not use codebooks")
            return
        missing = missing_codebooks(args.codebooks, args.m_subquantizers, args.nbits, N_LAYERS)
        bits = bits_per_dimension(args.m_subquantizers, args.nbits, args.head_dim)
        if missing:
            raise SystemExit(f"missing {len(missing)} codebooks, first: {missing[0]}")
        print(
            f"codebooks ok preset={args.preset} M={args.m_subquantizers} "
            f"nbits={args.nbits} bits/dim={bits:g} root={args.codebooks}"
        )
        return
    ENGINE = VecInferEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
