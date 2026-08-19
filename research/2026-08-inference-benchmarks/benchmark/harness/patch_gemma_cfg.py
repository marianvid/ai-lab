import json, shutil, sys, os
# The model config declares per-layer head_dim; vLLM wants the global value to
# size attention buffers, which is a legitimate read. Tell transformers so.
for d in ["/models/nvfp4/gemma-4-26b-a4b", "/models/nvfp4/gemma-4-12b"]:
    p = os.path.join(d, "config.json")
    if not os.path.exists(p):
        print("lipseste", p); continue
    if not os.path.exists(p + ".orig"):
        shutil.copy2(p, p + ".orig")
    c = json.load(open(p))
    for key in ("text_config", "vision_config", "audio_config"):
        if isinstance(c.get(key), dict):
            c[key]["allow_global_per_layer_attribute_access"] = True
    c["allow_global_per_layer_attribute_access"] = True
    json.dump(c, open(p, "w"), indent=2)
    print("marcat", p)
