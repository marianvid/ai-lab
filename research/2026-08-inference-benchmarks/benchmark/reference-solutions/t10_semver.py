import re
_CORE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
def _parse(v):
    if not isinstance(v, str) or not v:
        raise ValueError("bad version")
    v = v.split("+", 1)[0]
    core, _, pre = v.partition("-")
    m = _CORE.match(core)
    if not m:
        raise ValueError("bad version")
    nums = tuple(int(x) for x in m.groups())
    if not pre:
        return nums, None
    ids = pre.split(".")
    if any(p == "" for p in ids):
        raise ValueError("bad prerelease")
    return nums, ids
def _cmp_pre(a, b):
    for x, y in zip(a, b):
        xn, yn = x.isdigit(), y.isdigit()
        if xn and yn:
            if int(x) != int(y): return -1 if int(x) < int(y) else 1
        elif xn != yn:
            return -1 if xn else 1
        elif x != y:
            return -1 if x < y else 1
    if len(a) == len(b): return 0
    return -1 if len(a) < len(b) else 1
def compare_versions(a, b):
    na, pa = _parse(a); nb, pb = _parse(b)
    if na != nb: return -1 if na < nb else 1
    if pa is None and pb is None: return 0
    if pa is None: return 1
    if pb is None: return -1
    return _cmp_pre(pa, pb)
