"""Fix vLLM's Gemma4 layer sizing under transformers >= 5.15.

Gemma-4 mixes two attention shapes: 25 sliding layers with head_dim 256 and 8
key-value heads, and 5 full-attention layers with head_dim 512 and 2 key-value
heads. Older transformers exposed the second set as `global_head_dim` and
`num_global_key_value_heads`, which is what vLLM reads. Transformers 5.15 drops
both names and publishes the authoritative values per layer instead, so vLLM's
lookups silently fall back to the sliding-layer numbers and then fail to load
the full-attention weights. Prefer the per-layer values when they exist.
"""
import os, shutil, sys

G = "/opt/ai/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/gemma4.py"
if not os.path.exists(G + ".orig"):
    shutil.copy2(G, G + ".orig")
src = open(G + ".orig").read()          # patch the pristine copy, drop the debug print

OLD_HEAD = '''        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim
'''
NEW_HEAD = '''        # transformers >= 5.15 no longer exposes global_head_dim; the real
        # per-layer value lives in config.per_layer_config[i].
        _per_layer = getattr(config, "per_layer_config", None)
        _lc = (
            _per_layer[layer_idx]
            if _per_layer is not None and layer_idx < len(_per_layer)
            else None
        )
        if _lc is not None and getattr(_lc, "head_dim", None):
            head_dim = _lc.head_dim
        elif self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim
'''

OLD_KV = '''        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads
'''
NEW_KV = '''        # Same story for the key-value head count: full-attention layers use a
        # different number, and only the per-layer config still carries it.
        if _lc is not None and getattr(_lc, "num_key_value_heads", None):
            num_kv_heads = _lc.num_key_value_heads
        elif use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads
'''

for old, new, what in ((OLD_HEAD, NEW_HEAD, "head_dim"), (OLD_KV, NEW_KV, "num_kv_heads")):
    if old not in src:
        print(f"EȘEC: nu găsesc blocul {what}"); sys.exit(1)
    src = src.replace(old, new, 1)

open(G, "w").write(src)
print("reparație aplicată pentru head_dim și num_kv_heads")
