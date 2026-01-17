from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx


@dataclass(frozen=True)
class Prompt:
    path: Path
    text: str
    sha256: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp OpenAI-compatible endpoints across model variants.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model prefix to benchmark (e.g. qwen3-coder:30b).",
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Base host, e.g. http://10.0.0.31:8000",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per prompt and model (default: 5).",
    )
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Directory containing .txt prompt files.",
    )
    parser.add_argument(
        "--results",
        default="benchmarks/results.jsonl",
        help="Append-only JSONL results file.",
    )
    parser.add_argument(
        "--responses-dir",
        default="benchmarks/responses",
        help="Directory to save model responses (requires --save-responses).",
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Persist full model responses to disk.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Request timeout in seconds (default: 1800).",
    )
    return parser.parse_args(argv)


def normalize_host(host: str) -> str:
    return host.rstrip("/")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts(prompts_dir: Path) -> list[Prompt]:
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompt_paths = sorted(prompts_dir.glob("*.txt"))
    if not prompt_paths:
        raise FileNotFoundError(f"No .txt prompts found in: {prompts_dir}")

    prompts: list[Prompt] = []
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        prompts.append(Prompt(path=path, text=text, sha256=sha256_text(text)))
    return prompts


def extract_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return list(payload)
        if all(isinstance(item, dict) for item in payload):
            return [item.get("id") for item in payload if item.get("id")]

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return extract_model_ids(payload["data"])
        if "models" in payload and isinstance(payload["models"], list):
            return extract_model_ids(payload["models"])

    return []


def fetch_models(client: httpx.Client, host: str) -> list[str]:
    base = normalize_host(host)
    endpoints = [f"{base}/models", f"{base}/v1/models"]
    last_error: Exception | None = None
    for url in endpoints:
        try:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            model_ids = extract_model_ids(data)
            if model_ids:
                return model_ids
        except Exception as exc:  # noqa: BLE001 - best-effort fallback across endpoints
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch models from server.")


def filter_models(model_ids: Iterable[str], prefix: str) -> list[str]:
    return sorted([model_id for model_id in model_ids if model_id.startswith(prefix)])


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_completed(results_path: Path) -> dict[tuple[str, str, int], str]:
    completed: dict[tuple[str, str, int], str] = {}
    if not results_path.exists():
        return completed

    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != "ok":
            continue
        model = record.get("model")
        prompt_file = record.get("prompt_file")
        prompt_hash = record.get("prompt_hash")
        run_index = record.get("run_index")
        if run_index is None:
            run_index = 1
        if model and prompt_file and prompt_hash:
            completed[(model, prompt_file, int(run_index))] = prompt_hash
    return completed


def append_result(results_path: Path, record: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()


def save_response(responses_dir: Path, model: str, prompt_file: Path, content: str) -> Path:
    model_slug = safe_slug(model) or "model"
    prompt_slug = safe_slug(prompt_file.stem) or "prompt"
    output_dir = responses_dir / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prompt_slug}.txt"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def truncate_response(text: str, limit: int = 103) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:50]}...{text[-50:]}"


def run_prompt(
    client: httpx.Client,
    host: str,
    model: str,
    prompt: Prompt,
) -> tuple[dict[str, Any], str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt.text}],
        "cache_prompt": False,
    }

    url = f"{normalize_host(host)}/v1/chat/completions"
    start = time.perf_counter()
    response = client.post(url, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    data = response.json()

    content = None
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")

    data["_elapsed_ms"] = round(elapsed_ms, 3)
    return data, content


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    prompts_dir = Path(args.prompts_dir)
    results_path = Path(args.results)
    responses_dir = Path(args.responses_dir)

    prompts = load_prompts(prompts_dir)
    completed = load_completed(results_path)

    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(timeout=timeout) as client:
        model_ids = fetch_models(client, args.host)
        targets = filter_models(model_ids, args.model)

        if not targets:
            print(f"No models matched prefix: {args.model}")
            return 1

        total_runs = len(targets) * len(prompts) * args.runs
        run_index = 0

        for model in targets:
            for prompt in prompts:
                for iteration in range(1, args.runs + 1):
                    run_index += 1
                    key = (model, prompt.path.name, iteration)
                    prompt_hash = prompt.sha256
                    if key in completed and completed[key] == prompt_hash:
                        print(
                            f"[{run_index}/{total_runs}] skip {model} "
                            f"{prompt.path.name} run {iteration}"
                        )
                        continue

                    started_at = datetime.now(timezone.utc).isoformat()
                    print(
                        f"[{run_index}/{total_runs}] run {model} "
                        f"{prompt.path.name} run {iteration}"
                    )

                    record: dict[str, Any] = {
                        "timestamp": started_at,
                        "host": normalize_host(args.host),
                        "model": model,
                        "prompt_file": prompt.path.name,
                        "prompt_hash": prompt_hash,
                        "prompt_chars": len(prompt.text),
                        "run_index": iteration,
                        "status": "error",
                    }

                    try:
                        data, content = run_prompt(
                            client,
                            args.host,
                            model,
                            prompt,
                        )
                        record["status"] = "ok"
                        record["response_id"] = data.get("id")
                        record["usage"] = data.get("usage")
                        record["timings"] = data.get("timings")
                        record["elapsed_ms"] = data.get("_elapsed_ms")
                        if content is not None:
                            record["response_chars"] = len(content)
                            record["response_preview"] = truncate_response(content)
                        if args.save_responses and content is not None:
                            response_path = save_response(
                                responses_dir,
                                model,
                                prompt.path,
                                content,
                            )
                            record["response_path"] = str(response_path)
                    except Exception as exc:  # noqa: BLE001 - capture failures per run
                        record["error"] = str(exc)

                    append_result(results_path, record)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
