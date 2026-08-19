"""How many characters per token, per language, for the candidate models."""
import json, collections
from transformers import AutoTokenizer
items=[json.loads(l) for l in open("/opt/ai/tools/relevance_set.jsonl", encoding="utf-8")]
MODELS={"Gemma-4-26B":"/models/nvfp4/gemma-4-26b-a4b",
        "Qwen3.6-35B (productie)":"/models/nvfp4/qwen3-coder-30b-a3b"}
by={}
for it in items:
    by.setdefault(it["tld"],[]).append((it["title"] or "")+"\n"+(it["desc"] or ""))
for name,path in MODELS.items():
    try: tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e: print(name,"->",repr(e)[:80]); continue
    print(f"\n### {name}")
    print(f"{'limba':<8}{'texte':>7}{'caractere':>11}{'tokeni':>9}{'car/token':>11}")
    tot_c=tot_t=0
    for tld in ("ru","ua","lt","pl","com","news"):
        txts=by.get(tld,[])
        if not txts: continue
        c=sum(len(t) for t in txts)
        t=sum(len(tok.encode(x, add_special_tokens=False)) for x in txts)
        tot_c+=c; tot_t+=t
        print(f"{tld:<8}{len(txts):>7}{c:>11}{t:>9}{c/t:>11.2f}")
    print(f"{'TOTAL':<8}{'':>7}{tot_c:>11}{tot_t:>9}{tot_c/tot_t:>11.2f}")
