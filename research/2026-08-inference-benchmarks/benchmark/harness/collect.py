#!/usr/bin/env python3
"""Gather every result file from the container into one JSON summary."""
import json, subprocess, sys
KEY="${AI_LAB_SSH_KEY:?}"; HOST="${AI_LAB_HOST:?}"

def sh(cmd):
    r = subprocess.run(["ssh","-i",KEY,HOST,f"LC_ALL=C pct exec 102 -- bash -c '{cmd}'"],
                       capture_output=True, text=True)
    return "\n".join(l for l in r.stdout.splitlines()
                     if "locale" not in l and "LC_" not in l and "LANG" not in l
                     and "are supported" not in l and "perl: warn" not in l)

names = [n for n in sh("ls /opt/ai/tools/*.json 2>/dev/null").split() if n.strip()]
out = {"speed": {}, "coding": {}, "classify": {}}
for path in names:
    base = path.split("/")[-1]
    raw = sh(f"cat {path}")
    try: d = json.loads(raw)
    except Exception: continue
    if base.startswith("sp-"):     out["speed"][base[3:-5]] = d
    elif base.startswith("code-"): out["coding"][base[5:-5]] = d
    elif base.startswith("cls-"):  out["classify"][base[4:-5]] = d
    elif base.startswith("res-"):  out["speed"][base[4:-5]] = d
json.dump(out, open(sys.argv[1] if len(sys.argv)>1 else "/dev/stdout","w"), ensure_ascii=False, indent=1)
print(f"speed={len(out['speed'])} coding={len(out['coding'])} classify={len(out['classify'])}", file=sys.stderr)
