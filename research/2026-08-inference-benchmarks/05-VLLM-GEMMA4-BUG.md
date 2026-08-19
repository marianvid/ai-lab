# A bug in vLLM 0.27.1 — Gemma-4 will not load

**Status: diagnosed, patched locally, not yet reported upstream.**

**Update, later the same day: vLLM nightly loads Gemma-4 with no patch at
all, and measures the same.** The fix exists upstream; this patch reproduces
it on the stable release. Keep it only if you are pinned to stable. Details
and the comparison are in `08-ENGINE-MODEL-SUPPORT.md`.

**If you upgrade vLLM, this patch disappears and Gemma-4 stops working.** Check
whether the upstream code still has the defect, and reapply if so.

## Symptom

Any Gemma-4 model in a native quantised format fails to start on vLLM 0.27.1.
Both variants tested fail, from different publishers and different quantisation
toolchains, so it is not a bad checkpoint:

- `nvidia/Gemma-4-26B-A4B-NVFP4` — modelopt, MoE
- `RedHatAI/gemma-4-12B-it-NVFP4` — compressed-tensors, dense

Two errors appear, one after the other. The second only becomes visible once the
first is worked around.

## Defect 1 — an exception caught by the wrong name

```
transformers.integrations.heterogeneity.configuration_utils.
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute...
```

Raised from `vllm/transformers_utils/model_arch_config_convertor.py:608`:

```python
head_dim = getattr(self.hf_text_config, "head_dim", 0)
```

The default argument shows the author expected the attribute to be *absent* on
some configs. Transformers 5.15 does not report it absent; it raises. And
`AmbiguousGlobalPerLayerAttributeError` subclasses **`RuntimeError`, not
`AttributeError`**, so `getattr`'s default does not catch it and startup dies.

**Workaround.** The exception message names its own escape hatch. Add one key to
the model's `config.json`, inside `text_config`:

```json
"allow_global_per_layer_attribute_access": true
```

`benchmark/harness/patch_gemma_cfg.py` does this and keeps a `.orig` backup. This is safe
here: vLLM wants the global value in order to size buffers for the largest case,
which is exactly the use the warning permits.

## Defect 2 — layers built with the wrong dimensions

Getting past defect 1 reveals the real problem:

```
AssertionError: Attempted to load weight (torch.Size([512]))
                into parameter (torch.Size([256]))
```

Gemma-4 mixes two attention shapes:

| Layer type | Count | `head_dim` | key-value heads |
|---|---:|---:|---:|
| sliding attention | 25 | 256 | 8 |
| full attention | 5 | **512** | **2** |

Older transformers published the second set under the names `global_head_dim`
and `num_global_key_value_heads`, and `vllm/model_executor/models/gemma4.py`
reads exactly those names:

```python
if self.is_full_attention:
    head_dim = getattr(config, "global_head_dim", config.head_dim)
else:
    head_dim = config.head_dim
```

**Transformers 5.15 removed both names** and publishes the authoritative values
per layer, in `config.per_layer_config[i]`. So `getattr` finds nothing, falls
back to the sliding-layer value, and every one of the 30 layers is built with
head_dim 256 and 8 key-value heads. Loading then fails on the first
full-attention layer.

Instrumented output confirming it — `global_head_dim` is `None` for all layers,
including the ones marked `full_attention`:

```
[DBG] layer=5  type=full_attention    chosen=256 cfg.head_dim=256 cfg.global_head_dim=None
[DBG] layer=11 type=full_attention    chosen=256 cfg.head_dim=256 cfg.global_head_dim=None
[DBG] layer=17 type=full_attention    chosen=256 cfg.head_dim=256 cfg.global_head_dim=None
```

## The checkpoint is fine

Before blaming vLLM, the safetensors headers were read directly and compared
against the config, layer by layer: **25 layers at 256, 5 at 512, zero
mismatches.** What transformers exposes also matches:

```
per_layer_config head_dim            {256: 25, 512: 5}
per_layer_config num_key_value_heads {8: 25, 2: 5}
```

## The patch

`benchmark/harness/fix_gemma4.py`. It prefers the per-layer value where transformers
provides one and keeps the old lookup as a fallback, so it is safe on older
transformers too. Two blocks changed in `gemma4.py`:

```python
_per_layer = getattr(config, "per_layer_config", None)
_lc = (_per_layer[layer_idx]
       if _per_layer is not None and layer_idx < len(_per_layer) else None)

if _lc is not None and getattr(_lc, "head_dim", None):
    head_dim = _lc.head_dim
elif self.is_full_attention:
    head_dim = getattr(config, "global_head_dim", config.head_dim)
else:
    head_dim = config.head_dim
```

and the same shape for `num_kv_heads`, reading `_lc.num_key_value_heads`.

Applied to
`/opt/ai/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/gemma4.py`,
with the pristine file kept as `gemma4.py.orig`.

## Verification — loading is not the same as working

A model built with wrong layer sizes could load and emit nonsense, so the fixed
model was compared against **the same model in GGUF on llama.cpp**, which never
had the bug:

| Measurement | NVFP4 / vLLM, patched | GGUF / llama.cpp | Difference |
|---|---:|---:|---:|
| chrF++ average | 69.68 | 69.50 | +0.18 |
| chrF++ Romanian | 68.68 | 67.90 | +0.78 |
| chrF++ French | 77.55 | 77.10 | +0.45 |
| chrF++ Ukrainian | 63.80 | 64.60 | −0.80 |
| Classification F1 | 0.969 | 0.968 | +0.001 |
| Coding | 10/10 | 10/10 | — |

Every difference sits inside measurement noise, across all seven languages. The
model computes correctly.

## Regression check — nothing else was disturbed

Only one file was modified, and it is loaded solely for Gemma-4 models. Verified
empirically on a non-Gemma model:

| | before the patch | after |
|---|---:|---:|
| Qwopus-27B, articles/s at c=8 | 17.64 | 17.67 |

## What it unlocks

| | before | after |
|---|---|---|
| Gemma-4 on vLLM | does not start | works |
| Throughput at c=32 | — | **159.6 articles/s**, ×17.6 over one request |
| Prompt reading | — | 15 611 tok/s |

Against the same model on llama.cpp — 8.1 articles/s at its best — this is a
twenty-fold difference in bulk throughput.

## Reporting upstream

Not yet done. It should be, because 0.27.1 is the current release and there is
no newer version to upgrade to. A report needs:

1. Minimal reproduction: `vllm serve nvidia/Gemma-4-26B-A4B-NVFP4` on
   transformers 5.15.
2. Both tracebacks.
3. The observation that `AmbiguousGlobalPerLayerAttributeError` derives from
   `RuntimeError`, which is why the `getattr` default does not shield the caller.
4. The patch above, as a pull request.

Worth checking first whether an issue already exists, and whether the `main`
branch has moved.

## After any vLLM upgrade

```bash
pct exec 102 -- ls -la /opt/ai/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/ | grep gemma4
# if gemma4.py.orig is gone, the package was replaced and the patch with it
pct exec 102 -- /opt/ai/vllm/.venv/bin/python /tmp/fix_gemma4.py   # from harness/
```

Then start a Gemma model and run one classification pass. If F1 lands near
0.969, the fix took.
