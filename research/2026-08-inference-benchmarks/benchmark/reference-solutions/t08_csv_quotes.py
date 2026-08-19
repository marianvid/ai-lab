def split_csv_line(line):
    fields, buf, i, n = [], [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            while True:
                if i >= n:
                    raise ValueError("unterminated quote")
                if line[i] == '"':
                    if i + 1 < n and line[i+1] == '"':
                        buf.append('"'); i += 2; continue
                    i += 1; break
                buf.append(line[i]); i += 1
        while i < n and line[i] != ",":
            buf.append(line[i]); i += 1
        fields.append("".join(buf)); buf = []
        if i >= n:
            return fields
        i += 1
