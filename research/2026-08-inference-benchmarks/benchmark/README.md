# benchmark/ — everything needed to reproduce the measurements

Self-contained. The scripts, the exact prompts, the evaluation data, and the
reference answers used to validate the coding test. Nothing here depends on the
rest of this directory; the documents one level up interpret what comes out.

```
harness/              the four measurement scripts, plus launchers
prompts/              the exact text sent to every model
goldensets/           the evaluation data, with provenance
reference-solutions/  Claude Opus 5 answers to the ten coding tasks
```

## The four measurements

| Script | Measures | Data it needs |
|---|---|---|
| `bench.py` | prefill, decode, time to first token, at three prompt sizes | none, builds its own |
| `bench_coding.py` | ten Python tasks, executed against hidden tests | `coding_tasks.json` |
| `bench_classify.py` | relevance F1 per language, and throughput | `relevance_set.jsonl` |
| `bench_translate.py` | 42 translations scored with chrF++ | `translation_set.jsonl` |

All four speak the OpenAI-compatible API, so the same script measures vLLM and
llama.cpp without change. By convention here vLLM answers on port 8098 and
llama.cpp on 8099.

## The prompts

Extracted verbatim from the scripts, so they cannot drift from what was actually
sent.

| File | Used by |
|---|---|
| `prompts/classification-system.txt` | the system message for the relevance test |
| `prompts/classification-user.txt` | how the five items per request are laid out |
| `prompts/translation.txt` | the translation instruction |
| `prompts/latency-prompt.txt` | the three growing code prompts |
| `prompts/coding-tasks.md` | all ten tasks and their hidden tests, readable |
| `prompts/coding_tasks.json` | the same, machine-readable — this is what runs |

The classification prompt is adapted from the ParallaxVox production prompt
`phase4_pass1_system.md`, with one deliberate change: production shows only the
headline, this test also shows the description, because the ground truth was
labelled on headline plus description. **Scores here therefore do not transfer
directly to production numbers.**

## Running it

```bash
# start a server
bash harness/run_vllm.sh /models/nvfp4/<model> 32768 0.90 --max-num-seqs 32
bash harness/run_gguf.sh /models/gguf/<model>.gguf 32768 '--parallel 8'

# measure
python harness/bench.py           http://127.0.0.1:8098 <label> out.json
python harness/bench_coding.py    http://127.0.0.1:8098 <label> out.json
python harness/bench_classify.py  http://127.0.0.1:8098 <label> <concurrency> 5 out.json
python harness/bench_translate.py http://127.0.0.1:8098 <label> 8 out.json
```

Needs `huggingface_hub` and `sacrebleu` in the Python environment, and `setpriv`
to run generated code as an unprivileged user.

Two data files must sit where the scripts expect them —
`/opt/ai/tools/relevance_set.jsonl` and `/opt/ai/tools/translation_set.jsonl` —
along with `coding_tasks.json`. Copy them from `goldensets/` and `prompts/`.

## Validate the test before trusting a score

```bash
# every reference solution must pass
for f in reference-solutions/*.py; do ... ; done
```

All ten pass. They exist because an evaluation with wrong tests is worse than no
evaluation. If a model scores badly, run these first.

## Behaviour that is not obvious

Each script asks the server to skip its reasoning pass
(`chat_template_kwargs: {"enable_thinking": false}`) and retries without the
argument if the server rejects it; strips any `<think>` block that survives; and
runs one untimed warm-up before timing.

Every one of those exists because leaving it out produced a confidently wrong
number. The five stories are in `../04-PITFALLS.md`, and they are worth reading
before adding a fifth measurement.

## Contributing a card

The point of publishing this is that no two hardware reviews measure the same
thing. Run these four tests on your GPU, keep the output JSON, and the numbers
become comparable with everything in `../02-RESULTS.md`.

Record alongside them: the card, its VRAM, the engine and version, the exact
model files, and the flags you started the server with.
