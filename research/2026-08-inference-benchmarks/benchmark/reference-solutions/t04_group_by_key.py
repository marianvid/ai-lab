def group_consecutive(items, key):
    out = []
    for it in items:
        k = key(it)
        if out and out[-1][0] == k:
            out[-1][1].append(it)
        else:
            out.append((k, [it]))
    return out
