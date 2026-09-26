# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
"""Validity check of a trussnet JMesh: tet orientation / degeneracy, face
manifoldness (every face in <= 2 tets), and the label-interface area per label
pair (compare two meshes: the interfaces must be the same surface).

usage: python3 tools/tncheck.py mesh.jmsh [other.jmsh]
"""
import base64, json, os, resource, sys, zlib
import numpy as np
_cap = int(float(os.environ.get('TNPLOT_MAX_GB', '6')) * (1 << 30))
resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))


def jarray(o):
    raw = zlib.decompress(base64.b64decode(o['_ArrayZipData_']))
    return np.frombuffer(raw, dtype=o['_ArrayType_']).reshape(o['_ArraySize_'])


def check(path):
    d = json.load(open(path))
    P = jarray(d['MeshNode']).astype(float)
    E = jarray(d['MeshElem'])
    T = E[:, :4].astype(np.int64) - (1 if E[:, :4].min() == 1 else 0)
    lab = E[:, 4]
    a, b, c, e = (P[T[:, k]] for k in range(4))
    v6 = np.einsum('ij,ij->i', np.cross(b - a, c - a), e - a)
    f = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]])
    fl = np.tile(lab, 4)
    key = np.sort(f, axis=1)
    o = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    ks = key[o]
    brk = np.r_[True, np.any(ks[1:] != ks[:-1], axis=1)]
    grp = np.cumsum(brk) - 1
    cnt = np.bincount(grp)
    # interface faces: shared by two tets of different labels; area per label pair
    pair_first = np.where(brk)[0]
    two = cnt == 2
    i0 = o[pair_first[two]]
    i1 = o[pair_first[two] + 1]
    diff = fl[i0] != fl[i1]
    fa = f[i0[diff]]
    area = 0.5 * np.linalg.norm(np.cross(P[fa[:, 1]] - P[fa[:, 0]], P[fa[:, 2]] - P[fa[:, 0]]), axis=1)
    lp = np.sort(np.stack([fl[i0[diff]], fl[i1[diff]]], 1), 1)
    areas = {}
    for (x, y), ar in zip(map(tuple, lp), area):
        areas[(x, y)] = areas.get((x, y), 0.0) + ar
    bnd = cnt == 1
    ext_area = 0.5 * np.linalg.norm(np.cross(P[f[o[pair_first[bnd]]]][:, 1] - P[f[o[pair_first[bnd]]]][:, 0],
                                             P[f[o[pair_first[bnd]]]][:, 2] - P[f[o[pair_first[bnd]]]][:, 0]), axis=1).sum()
    # radius-edge ratio (circumradius / shortest edge; regular tet 0.612, TetGen -q 2)
    u, v, w = b - a, c - a, e - a
    num = (np.einsum('ij,ij->i', u, u)[:, None] * np.cross(v, w) + np.einsum('ij,ij->i', v, v)[:, None] * np.cross(w, u)
           + np.einsum('ij,ij->i', w, w)[:, None] * np.cross(u, v))
    R = np.linalg.norm(num, axis=1) / np.maximum(np.abs(2 * np.einsum('ij,ij->i', u, np.cross(v, w))), 1e-300)
    emin = np.min(np.stack([np.linalg.norm(x, axis=1) for x in (u, v, w, c - b, e - b, e - c)], 1), 1)
    rr = R / emin
    print(f'   radius-edge: median {np.median(rr):.3f}, p99 {np.percentile(rr, 99):.2f}, max {rr.max():.3g}; '
          f'> 2: {np.sum(rr > 2)} ({100 * np.mean(rr > 2):.3f}%), > 1.5: {np.sum(rr > 1.5)} ({100 * np.mean(rr > 1.5):.3f}%)')
    print(f'{path}: {len(P)} nodes {len(T)} tets | orientation: {np.sum(v6 > 0)} +, {np.sum(v6 < 0)} -, '
          f'{np.sum(np.abs(v6) < 1e-12)} degenerate | faces in >2 tets: {np.sum(cnt > 2)} | '
          f'exterior area {ext_area:.1f}, interface area {sum(areas.values()):.1f}')
    return areas


A = check(sys.argv[1])
if len(sys.argv) > 2:
    B = check(sys.argv[2])
    worst = max([abs(A.get(k, 0) - B.get(k, 0)) / max(1e-9, A.get(k, 0)) for k in set(A) | set(B)] or [0.0])
    print(f"interface area per label pair: max relative difference {100 * worst:.3f}%")
    for k in sorted(set(A) | set(B)):
        print(f"   pair {k}: {A.get(k, 0):.1f} -> {B.get(k, 0):.1f}")
