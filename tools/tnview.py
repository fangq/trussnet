"""Read trussnet debug dumps (JSON header line + raw arrays)."""
import json, numpy as np

def load(path):
    with open(path, 'rb') as f:
        head = json.loads(f.readline())
        base = f.tell()
        buf = f.read()
    out = {}
    for name, (dt, shape, off) in head.items():
        cnt = int(np.prod(shape))
        out[name] = np.frombuffer(buf, dtype=dt, count=cnt, offset=off).reshape(shape)
    return out
