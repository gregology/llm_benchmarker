# llm-benchmarker

Benchmark llama.cpp OpenAI-compatible endpoints across multiple model variants.

## Quick start

```bash
uv run benchmark --model "qwen3-coder:30b" --host "http://10.0.0.31:8000"
```

Prompts live in `prompts/`. Results append to `benchmarks/results.jsonl` by default.
Use `--runs` to control how many times each model+prompt pair is executed (default: 5).
Prompt caching is disabled by default (llama.cpp `cache_prompt=false`).
