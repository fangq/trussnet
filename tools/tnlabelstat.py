"""Per-label tet count, volume and mean tet volume of a trussnet JMesh.
usage: python3 tools/tnlabelstat.py mesh.jmsh"""
import base64, json, os, resource, sys, zlib
import numpy as np
_cap = int(float(os.environ.get('TNPLOT_MAX_GB', '6')) * (1 << 30))
resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))
d = json.load(open(sys.argv[1]))
ja = lambda o: np.frombuffer(zlib.decompress(base64.b64decode(o['_ArrayZipData_'])), dtype=o['_ArrayType_']).reshape(o['_ArraySize_'])
P = ja(d['MeshNode']).astype(float); E = ja(d['MeshElem']); T = E[:, :4].astype(np.int64); T -= T.min(); lab = E[:, 4]
a, b, c, e = (P[T[:, k]] for k in range(4))
vol = np.abs(np.einsum('ij,ij->i', np.cross(b - a, c - a), e - a)) / 6
print(f'{len(P)} nodes, {len(T)} tets')
for l in np.unique(lab):
    k = lab == l
    print(f'LABEL {l}: {k.sum()} tets, volume {vol[k].sum():.1f} mm3, mean tet volume {vol[k].mean():.3f} mm3')
