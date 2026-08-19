#!/usr/bin/env python3
"""Multilingual relevance classification: quality AND throughput in one run.

Real news headlines in Russian, Ukrainian, Lithuanian, Polish, Romanian and
more, each already judged by a human-reviewed reference. The model gets a small
batch per request; many requests run at once so we can see what continuous
batching actually buys.

Usage: bench_classify.py <base_url> <label> <concurrency> [batch] [out.json]
"""
import json, re, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE   = sys.argv[1]
LABEL  = sys.argv[2]
CONC   = int(sys.argv[3])
BATCH  = int(sys.argv[4]) if len(sys.argv) > 4 else 5
OUT    = sys.argv[5] if len(sys.argv) > 5 else None
DATA   = "/opt/ai/tools/relevance_set.jsonl"

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
def strip_think(t):
    """Reasoning models wrap their scratch work in <think> tags; drop it."""
    t = THINK_RE.sub("", t)
    if "</think>" in t:            # truncated opening tag
        t = t.split("</think>", 1)[1]
    return t.strip()

def post_json(url, payload):
    """Ask once with thinking switched off; if the server rejects that
    argument, ask again without it."""
    body = dict(payload)
    body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in (body, payload):
        req = urllib.request.Request(url, data=json.dumps(attempt).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and attempt is body:
                continue
            raise
    raise RuntimeError("both attempts failed")


SYSTEM = """You classify news items for relevance to: Ukraine-Russia War. For each item you see the headline and a short description. Decide:
relevant (bool): is this about or meaningfully connected to the conflict/topic? This includes NOT JUST combat, but also:
- Sanctions on a party's goods/officials/institutions (even oddly-specific ones, e.g. sanctions on a food product)
- Alliance/bloc diplomatic meetings, summits, or statements involving the conflict parties, even if one isn't named
- Cultural, media, or sporting bans/restrictions tied to a conflict party
- Domestic administrative/security actions that could relate to the conflict (airport restrictions, special regimes, military recruitment/education changes)
- Attacks, threats, or legal cases involving a party's state media figures or officials
Items may be in any language.
The items are CURRENT NEWS, more recent than your training data. A name, official title, or event you do not recognize is most likely real and simply newer than your knowledge — never mark an item irrelevant because a name or role looks unfamiliar. Judge relevance to the topic, not whether you recognize the facts.
Respond ONLY with a JSON array, one object per item, no markdown fences:
[{"idx": 0, "relevant": true}, {"idx": 1, "relevant": false}]"""

items = [json.loads(l) for l in open(DATA, encoding="utf-8")]
batches = [items[i:i+BATCH] for i in range(0, len(items), BATCH)]

def model_name():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]
MODEL = model_name()

lock = threading.Lock()
stats = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "failed": 0}
preds = {}

def parse_array(text, n):
    """Pull the JSON array out of whatever the model wrapped it in."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("["): t = p; break
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e == -1: return None
    try:
        arr = json.loads(t[s:e+1])
    except json.JSONDecodeError:
        return None
    out = {}
    for o in arr:
        if isinstance(o, dict) and "idx" in o and "relevant" in o:
            out[int(o["idx"])] = bool(o["relevant"])
    return out

def run_batch(bi_batch):
    bi, batch = bi_batch
    lines = []
    for j, it in enumerate(batch):
        d = (it["desc"] or "")[:400]
        lines.append(f'{{"idx": {j}, "title": {json.dumps(it["title"], ensure_ascii=False)}, "description": {json.dumps(d, ensure_ascii=False)}}}')
    user = "Classify these items:\n" + "\n".join(lines)
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}],
               "max_tokens": 1200, "temperature": 0.0}
    try:
        obj = post_json(BASE + "/v1/chat/completions", payload)
    except Exception:
        with lock: stats["failed"] += 1
        return
    txt = strip_think(obj["choices"][0]["message"]["content"] or "")
    u = obj.get("usage", {})
    got = parse_array(txt, len(batch)) or {}
    with lock:
        stats["prompt_tokens"]     += u.get("prompt_tokens", 0)
        stats["completion_tokens"] += u.get("completion_tokens", 0)
        stats["requests"] += 1
        if not got: stats["failed"] += 1
        for j, it in enumerate(batch):
            if j in got: preds[it["url"]] = got[j]

print(f"\n=== {LABEL} | {MODEL} | concurrency={CONC} batch={BATCH} ===", flush=True)

# The first pass after a server start runs about 40% slow while the kernel
# autotune cache settles. Burn one untimed pass over a slice of the data.
_warm = batches[: max(2, CONC)]
with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(run_batch, enumerate(_warm)))
with lock:
    stats.update({"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "failed": 0})
    preds.clear()

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(run_batch, enumerate(batches)))
wall = time.perf_counter() - t0

tp = fp = tn = fn = 0
per_lang = {}
for it in items:
    p = preds.get(it["url"])
    if p is None: continue
    g = it["gold"]
    if   g and p:         tp += 1; k = "tp"
    elif not g and p:     fp += 1; k = "fp"
    elif not g and not p: tn += 1; k = "tn"
    else:                 fn += 1; k = "fn"
    d = per_lang.setdefault(it["tld"], {"tp":0,"fp":0,"tn":0,"fn":0})
    d[k] += 1

def scores(d):
    tp,fp,tn,fn = d["tp"],d["fp"],d["tn"],d["fn"]
    n = tp+fp+tn+fn
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {"n":n, "accuracy":round((tp+tn)/n,4) if n else 0,
            "precision":round(prec,4), "recall":round(rec,4), "f1":round(f1,4)}

overall = scores({"tp":tp,"fp":fp,"tn":tn,"fn":fn})
res = {
  "label": LABEL, "model": MODEL, "concurrency": CONC, "batch": BATCH,
  "wall_s": round(wall,2),
  "answered": len(preds), "of": len(items),
  "requests": stats["requests"], "failed_requests": stats["failed"],
  "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
  "prefill_tok_s": round(stats["prompt_tokens"]/wall,1),
  "decode_tok_s":  round(stats["completion_tokens"]/wall,1),
  "items_per_s":   round(len(preds)/wall,2),
  "quality": overall,
  "per_language": {k: scores(v) for k,v in sorted(per_lang.items(), key=lambda x:-x[1]["tp"]-x[1]["tn"]-x[1]["fp"]-x[1]["fn"])},
}
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
if OUT:
    with open(OUT,"w") as f: json.dump(res,f,ensure_ascii=False,indent=2)
