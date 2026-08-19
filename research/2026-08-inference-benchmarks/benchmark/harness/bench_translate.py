#!/usr/bin/env python3
"""Translate six real articles into seven languages and score them.

Scoring is deliberately mechanical. chrF++ compares the model's translation to
the reference character-n-gram by character-n-gram; it needs no judge model and
cannot play favourites. Alongside it, two checks that catch the failures a score
can hide: English left untranslated, and numbers that changed.

Usage: bench_translate.py <base_url> <label> <concurrency> [out.json]
"""
import json, re, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from sacrebleu.metrics import CHRF

BASE  = sys.argv[1]
LABEL = sys.argv[2]
CONC  = int(sys.argv[3]) if len(sys.argv) > 3 else 8
OUT   = sys.argv[4] if len(sys.argv) > 4 else None
DATA  = "/opt/ai/tools/translation_set.jsonl"

LANGS = {"ro":"Romanian","de":"German","fr":"French","es":"Spanish",
         "pl":"Polish","uk":"Ukrainian","ru":"Russian"}
chrf = CHRF(word_order=2)          # chrF++

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
def strip_think(t):
    t = THINK_RE.sub("", t or "")
    if "</think>" in t: t = t.split("</think>", 1)[1]
    return t.strip()

def post_json(url, payload):
    body = dict(payload); body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in (body, payload):
        req = urllib.request.Request(url, data=json.dumps(attempt).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and attempt is body: continue
            raise
    raise RuntimeError("both attempts failed")

def model_name():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]
MODEL = model_name()

items = [json.loads(l) for l in open(DATA, encoding="utf-8")]
lock = threading.Lock()
results = []
usage_total = {"prompt": 0, "completion": 0}

# words that are ordinary English but would be wrong to leave in a translation
EN_MARKERS = re.compile(r"\b(the|and|of|with|that|said|which|from|have|been|were|their)\b", re.I)
# Languages write 4,475 as 4.475 or 4 475. Strip separators that sit between
# digits, then compare bare digit runs, so formatting is not mistaken for error.
SEP_BETWEEN_DIGITS = re.compile(r"(?<=\d)[.,\u00a0\u202f\u2009 ](?=\d)")
DIGITS = re.compile(r"\d+")
def numbers_in(text):
    return set(DIGITS.findall(SEP_BETWEEN_DIGITS.sub("", text or "")))

def fence_strip(t):
    if "```" in t:
        parts = [p for p in t.split("```") if p.strip()]
        if parts:
            p = parts[0].strip()
            if p.lower().startswith(("markdown", "md")): p = p.split("\n", 1)[-1]
            return p.strip()
    return t

def one(item):
    lang = LANGS[item["lang"]]
    prompt = (f"Translate the article below into {lang}.\n\n"
              f"Keep the Markdown structure. Keep every number, name, place and quoted "
              f"phrase faithful to the original. Produce natural, publishable {lang} — "
              f"this is for a finished article, not a literal gloss.\n\n"
              f"Output only the translation, with no preamble and no commentary.\n\n"
              f"---\n{item['source']}")
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 4000, "temperature": 0.0}
    t0 = time.perf_counter()
    try:
        obj = post_json(BASE + "/v1/chat/completions", payload)
    except Exception as e:
        with lock: results.append({"slug": item["slug"], "lang": item["lang"],
                                   "error": repr(e)[:160]})
        return
    hyp = fence_strip(strip_think(obj["choices"][0]["message"]["content"]))
    u = obj.get("usage", {})
    ref = item["reference"]
    score = chrf.sentence_score(hyp, [ref]).score
    # mechanical checks
    en_hits = len(EN_MARKERS.findall(hyp)) if item["lang"] != "en" else 0
    missing_nums = len(numbers_in(item["source"]) - numbers_in(hyp))
    with lock:
        usage_total["prompt"]     += u.get("prompt_tokens", 0)
        usage_total["completion"] += u.get("completion_tokens", 0)
        results.append({"slug": item["slug"], "lang": item["lang"],
                        "chrf": round(score, 2),
                        "english_words_left": en_hits,
                        "numbers_missing": missing_nums,
                        "out_chars": len(hyp),
                        "ref_chars": len(ref),
                        "seconds": round(time.perf_counter() - t0, 1),
                        "text": hyp})

print(f"\n=== TRANSLATE | {LABEL} | {MODEL} | conc={CONC} ===", flush=True)
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(one, items))
wall = time.perf_counter() - t0

ok = [r for r in results if "chrf" in r]
by_lang = {}
for r in ok:
    by_lang.setdefault(r["lang"], []).append(r["chrf"])
summary = {
  "label": LABEL, "model": MODEL, "concurrency": CONC,
  "wall_s": round(wall, 1), "translated": len(ok), "of": len(items),
  "failed": len(results) - len(ok),
  "chrf_mean": round(sum(r["chrf"] for r in ok) / len(ok), 2) if ok else 0,
  "chrf_by_lang": {k: round(sum(v)/len(v), 2) for k, v in sorted(by_lang.items())},
  "english_left_total": sum(r["english_words_left"] for r in ok),
  "numbers_missing_total": sum(r["numbers_missing"] for r in ok),
  "prompt_tokens": usage_total["prompt"], "completion_tokens": usage_total["completion"],
  "decode_tok_s": round(usage_total["completion"] / wall, 1) if wall else 0,
}
print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({**summary, "items": results}, f, ensure_ascii=False, indent=2)
