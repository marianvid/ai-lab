def flatten(obj, sep="."):
    out = {}
    def walk(node, path):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list) and node:
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
        else:
            out[sep.join(path)] = node
    walk(obj, [])
    return out
