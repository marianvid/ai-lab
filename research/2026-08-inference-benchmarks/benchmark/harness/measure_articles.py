#!/usr/bin/env python3
"""How long is a real article, in characters and in tokens?

Samples the ParallaxVox pool, fetches with trafilatura and no character cap,
and reports the distribution. Links are from July; many will be gone, which is
itself worth measuring.
"""
import json, random, re, statistics, sys, collections
import concurrent.futures as cf
import requests, trafilatura

POOL = "/Volumes/Marian_Backup/work/parallaxvox/benchmark-data/gold/relevance/pool.json"
PER_TLD = 30
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

pool = json.load(open(POOL))
by = collections.defaultdict(list)
for a in pool:
    m = re.search(r'https?://([^/]+)', a.get("url", "") or "")
    if m: by[m.group(1).split(".")[-1]].append(a)

random.seed(7)
sample = []
for tld in ("ru", "ua", "lt", "pl", "com", "news", "md", "de"):
    items = by.get(tld, [])
    sample += random.sample(items, min(PER_TLD, len(items)))
print(f"esantion: {len(sample)} adrese din {len(pool)}", file=sys.stderr)

def one(a):
    url = a["url"]
    tld = re.search(r'https?://([^/]+)', url).group(1).split(".")[-1]
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            return {"tld": tld, "ok": False, "why": f"http {r.status_code}"}
        txt = trafilatura.extract(r.text, include_comments=False,
                                  include_tables=False, favor_precision=True)
        if not txt or len(txt) < 200:
            return {"tld": tld, "ok": False, "why": "no text"}
        return {"tld": tld, "ok": True, "chars": len(txt),
                "paras": len([p for p in txt.split("\n") if p.strip()])}
    except Exception as e:
        return {"tld": tld, "ok": False, "why": type(e).__name__}

res = []
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i, r in enumerate(ex.map(one, sample), 1):
        res.append(r)
        if i % 25 == 0: print(f"  {i}/{len(sample)}", file=sys.stderr)

ok = [r for r in res if r["ok"]]
print(f"\nreusite: {len(ok)}/{len(res)}  ({100*len(ok)/len(res):.0f}%)")
why = collections.Counter(r["why"] for r in res if not r["ok"])
print("esecuri:", dict(why.most_common(6)))

def dist(vals, label):
    vals = sorted(vals); n = len(vals)
    if n < 3: return
    print(f"{label:<22} n={n:>4}  mediana={statistics.median(vals):>7.0f} "
          f"p75={vals[int(n*.75)]:>7.0f} p90={vals[int(n*.90)]:>7.0f} "
          f"p99={vals[int(min(n-1,n*.99))]:>7.0f} max={vals[-1]:>7.0f}")

print()
dist([r["chars"] for r in ok], "caractere, tot")
dist([r["paras"] for r in ok], "paragrafe, tot")
print()
for tld in ("ru", "ua", "lt", "pl", "com", "news"):
    v = [r["chars"] for r in ok if r["tld"] == tld]
    if len(v) >= 3: dist(v, f"caractere .{tld}")

json.dump(res, open("/private/tmp/claude-501/-Volumes-Marian-Backup-work-ai-lab/679898fe-5c9c-428d-994a-2a02a21e69ae/scratchpad/article_lengths.json","w"))
