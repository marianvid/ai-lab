import re
def parse_duration(s: str) -> int:
    s = re.sub(r"\s+", "", s)
    if not s:
        raise ValueError("empty duration")
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    total, pos = 0, 0
    pat = re.compile(r"(\d+)([a-zA-Z])")
    while pos < len(s):
        m = pat.match(s, pos)
        if not m:
            raise ValueError(f"bad duration at {pos}: {s!r}")
        num, unit = m.group(1), m.group(2).lower()
        if unit not in mult:
            raise ValueError(f"unknown unit {unit!r}")
        total += int(num) * mult[unit]
        pos = m.end()
    return total
