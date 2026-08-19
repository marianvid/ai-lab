def topo_sort(graph):
    deps = {n: set(d) for n, d in graph.items()}
    for d in list(graph.values()):
        for x in d:
            deps.setdefault(x, set())
    out = []
    while deps:
        free = sorted(n for n, d in deps.items() if not d)
        if not free:
            raise ValueError("cycle")
        n = free[0]
        out.append(n)
        del deps[n]
        for d in deps.values():
            d.discard(n)
    return out
