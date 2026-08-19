#!/usr/bin/env python3
"""Measure an OpenAI-compatible inference server the way a coding agent uses it.

Two numbers matter and they are different things:
  prefill  -- how fast the model reads the prompt (agent pastes whole files)
  decode   -- how fast the model writes the answer (agent waits for the patch)

Usage: bench.py <base_url> <label> [out.json]
"""
import json, sys, time, urllib.request

BASE, LABEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

def post(path, payload, stream=False):
    """Reasoning models otherwise spend the whole budget thinking, and stream
    that thinking on a separate field, which hides the real first token."""
    import urllib.error
    body = dict(payload); body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in (body, payload):
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(attempt).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        try:
            return urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and attempt is body:
                continue
            raise
    raise RuntimeError("both attempts failed")

def model_name():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]

MODEL = model_name()

# A prompt that looks like real agent work: a chunk of source plus a request.
UNIT = (
    "def process_record(record, config, cache):\n"
    "    key = record.get('id')\n"
    "    if key is None:\n"
    "        raise ValueError('record without id')\n"
    "    if key in cache:\n"
    "        return cache[key]\n"
    "    result = {'id': key, 'fields': {}}\n"
    "    for name, spec in config['fields'].items():\n"
    "        raw = record.get(name)\n"
    "        if raw is None and spec.get('required'):\n"
    "            raise KeyError(name)\n"
    "        result['fields'][name] = spec['parse'](raw) if raw is not None else None\n"
    "    cache[key] = result\n"
    "    return result\n\n"
)

def make_prompt(repeats):
    body = "".join(f"# --- module {i} ---\n" + UNIT for i in range(repeats))
    return (
        "Here is part of a Python codebase:\n\n```python\n" + body + "```\n\n"
        "Rewrite process_record so that parse failures are collected into a list "
        "of errors instead of raising on the first one. Return only the new function."
    )

def run(prompt, max_tokens, tag):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    usage = {}
    resp = post("/v1/chat/completions", payload, stream=True)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            dl = ch.get("delta", {})
            piece = dl.get("content") or dl.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_out += 1
    total = time.perf_counter() - t0
    p_in = usage.get("prompt_tokens")
    p_out = usage.get("completion_tokens", n_out)
    decode_s = total - (ttft or 0)
    return {
        "test": tag,
        "prompt_tokens": p_in,
        "output_tokens": p_out,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 2),
        "prefill_tok_s": round(p_in / ttft, 1) if (p_in and ttft) else None,
        "decode_tok_s": round((p_out - 1) / decode_s, 1) if decode_s > 0 and p_out > 1 else None,
    }

TESTS = [
    (make_prompt(2),   256, "short prompt (~500 tok)"),
    (make_prompt(60),  256, "medium prompt (~9k tok)"),
    (make_prompt(220), 256, "long prompt (~32k tok)"),
]

results = {"label": LABEL, "model": MODEL, "runs": []}
print(f"\n=== {LABEL} | {MODEL} ===", flush=True)
# one untimed warm-up so the first real number is not a cold start
try:
    run(make_prompt(2), 16, "warmup")
except Exception as e:
    print("warmup failed:", e, flush=True)

for prompt, mt, tag in TESTS:
    try:
        r = run(prompt, mt, tag)
    except Exception as e:
        r = {"test": tag, "error": repr(e)}
    results["runs"].append(r)
    print(json.dumps(r), flush=True)

if OUT:
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
