#!/usr/bin/env python3
"""Ask a model to solve ten Python tasks, then actually run its code.

Nothing is graded by opinion: each answer is executed against a hidden test
block and either passes or does not. The generated code runs as `nobody`, in a
throwaway directory, with a hard timeout.

Usage: bench_coding.py <base_url> <label> [out.json]
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request, urllib.error

BASE  = sys.argv[1]
LABEL = sys.argv[2]
OUT   = sys.argv[3] if len(sys.argv) > 3 else None
TASKS = json.load(open("/opt/ai/tools/coding_tasks.json", encoding="utf-8"))

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


def model_name():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]
MODEL = model_name()

def ask(prompt):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 6000, "temperature": 0.0}
    obj = post_json(BASE + "/v1/chat/completions", payload)
    txt = strip_think(obj["choices"][0]["message"]["content"] or "")
    return txt, obj.get("usage", {})

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
def extract_code(text):
    blocks = CODE_RE.findall(text)
    if blocks:
        return max(blocks, key=len)
    return text  # some models answer with bare code

def run_code(code, tests):
    prog = code + "\n\n" + tests + "\nprint('___PASS___')\n"
    d = tempfile.mkdtemp(prefix="codebench_", dir="/tmp")
    path = os.path.join(d, "prog.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prog)
    os.chmod(d, 0o777); os.chmod(path, 0o755)
    try:
        r = subprocess.run(["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
                            "python3", path],
                           capture_output=True, text=True, timeout=25, cwd=d)
        if "___PASS___" in r.stdout:
            return True, ""
        err = (r.stderr or r.stdout).strip().splitlines()
        return False, (err[-1][:180] if err else "no output")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, repr(e)[:180]

print(f"\n=== CODING | {LABEL} | {MODEL} ===", flush=True)
results, passed = [], 0
t_all = time.perf_counter()
for t in TASKS:
    t0 = time.perf_counter()
    try:
        text, usage = ask(t["prompt"])
    except Exception as e:
        results.append({"id": t["id"], "pass": False, "why": f"request failed: {e!r}"[:180]})
        print(f"  FAIL {t['id']}  request failed", flush=True)
        continue
    ok, why = run_code(extract_code(text), t["tests"])
    passed += ok
    results.append({"id": t["id"], "pass": ok, "why": why,
                    "gen_tokens": usage.get("completion_tokens"),
                    "seconds": round(time.perf_counter() - t0, 1)})
    print(f"  {'PASS' if ok else 'FAIL'} {t['id']}  {'' if ok else why}", flush=True)

res = {"label": LABEL, "model": MODEL,
       "passed": passed, "of": len(TASKS),
       "score": round(passed / len(TASKS), 3),
       "wall_s": round(time.perf_counter() - t_all, 1),
       "tasks": results}
print(json.dumps({k: v for k, v in res.items() if k != "tasks"}, indent=1), flush=True)
if OUT:
    with open(OUT, "w") as f: json.dump(res, f, ensure_ascii=False, indent=2)
