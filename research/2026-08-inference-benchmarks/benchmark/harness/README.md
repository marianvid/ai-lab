# The harness

These four scripts are the actual product of this work. The numbers are output;
the scripts are the definition of what was measured.

All of them speak the OpenAI-compatible API, so they work against vLLM
(port 8098 by convention here) and llama.cpp (8099) without change.

| Script | Measures |
|---|---|
| `bench.py` | latency: prefill, decode and time-to-first-token at three prompt sizes |
| `bench_coding.py` | ten Python tasks, executed against hidden tests |
| `bench_classify.py` | multilingual relevance: F1 per language, and throughput |
| `bench_translate.py` | 42 translations scored with chrF++, plus mechanical checks |

Launchers and helpers:

| Script | Purpose |
|---|---|
| `run_vllm.sh` | start a vLLM server: model, context, GPU fraction, extra flags |
| `run_gguf.sh` | start a llama.cpp server: file, context, extra flags |
| `timeload.sh` | time load and unload for either engine |
| `collect.py` | pull every result file out of the container into one JSON |
| `fix_gemma4.py` | the vLLM patch — see `../05-VLLM-GEMMA4-BUG.md` |
| `patch_gemma_cfg.py` | adds the one config key Gemma-4 needs |

`reference-solutions/` holds Claude Opus 5 answers to all ten coding tasks. All
ten pass. They exist to validate the test suite itself: an eval with wrong tests
is worse than no eval. Run them before trusting a low score from any model.

## Requirements inside the container

`/opt/ai/tools/.venv` with `huggingface_hub` and `sacrebleu`, plus `setpriv` for
running generated code as `nobody`. Data files `relevance_set.jsonl`,
`translation_set.jsonl` and `coding_tasks.json` must sit in `/opt/ai/tools/`.

## Behaviour worth knowing

Every script asks the server to skip its reasoning pass and falls back cleanly if
the server rejects the argument; strips `<think>` blocks that survive; and runs
an untimed warm-up before timing. Each of those exists because its absence
produced a wrong result — the stories are in `../04-PITFALLS.md`.
