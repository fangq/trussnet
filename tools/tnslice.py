# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
"""Cross-sections of a trussnet JMesh: the exact intersection of every tet with a
plane (a triangle or a quad), coloured by label, element edges drawn -- three
orthogonal slices through the mesh centre by default.

usage: python3 tools/tnslice.py mesh.jmsh out.png [--pos x,y,z] [--names a,b,..]
       [--zoom x0,x1,y0,y1 (fractions of the axial slice)] [--dpi N]
"""

import argparse
import base64
import json
import os
import resource
import zlib

import numpy as np

# self-imposed memory ceiling (TNPLOT_MAX_GB, default 6), as tnplot.py
_cap = int(float(os.environ.get("TNPLOT_MAX_GB", "6")) * (1 << 30))
resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def jarray(o):
    raw = zlib.decompress(base64.b64decode(o["_ArrayZipData_"]))
    return np.frombuffer(raw, dtype=o["_ArrayType_"]).reshape(o["_ArraySize_"])


def load(path):
    d = json.load(open(path))
    P = jarray(d["MeshNode"]).astype(np.float64)
    E = jarray(d["MeshElem"])
    T = E[:, :4].astype(np.int64)
    if T.min() == 1:
        T -= 1
    return P, T, E[:, 4].astype(np.int64)


EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def section(P, T, lab, axis, pos):
    """the polygons (2-D, in the other two axes) where the tets cross axis = pos"""
    c = P[:, axis][T] - pos  # (M, 4) signed distances
    c[c == 0] = 1e-12  # a node on the plane counts as above
    cross = (c.min(1) < 0) & (c.max(1) > 0)
    Ts, cs, ls = T[cross], c[cross], lab[cross]
    other = [a for a in range(3) if a != axis]
    Q = P[:, other]
    a, b = EDGES[:, 0], EDGES[:, 1]
    da, db = cs[:, a], cs[:, b]  # (n, 6)
    hit = da * db < 0
    t = np.where(hit, da / np.where(hit, da - db, 1), 0)
    pa, pb = Q[Ts[:, a]], Q[Ts[:, b]]  # (n, 6, 2)
    pts = pa + t[..., None] * (pb - pa)
    nh = hit.sum(1)  # 3 (triangle) or 4 (quad)
    polys, cols = [], []
    for k in (3, 4):
        sel = nh == k
        if not sel.any():
            continue
        p = pts[sel][hit[sel]].reshape(-1, k, 2)
        ctr = p.mean(1, keepdims=True)
        ang = np.arctan2(p[..., 1] - ctr[..., 1], p[..., 0] - ctr[..., 0])
        p = np.take_along_axis(p, np.argsort(ang, 1)[..., None], 1)  # convex order
        polys.extend(p)
        cols.extend(ls[sel])
    return polys, np.asarray(cols), other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("png")
    ap.add_argument(
        "--pos", default=None, help="x,y,z of the three slices (default: the mesh centre)"
    )
    ap.add_argument("--names", default=None, help="label names, comma separated (label 1 first)")
    ap.add_argument(
        "--zoom", default=None, help="x0,x1,y0,y1: an extra zoomed panel of the axial slice"
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--lw", type=float, default=0.08, help="element edge width")
    a = ap.parse_args()

    P, T, lab = load(a.mesh)
    lo, hi = P.min(0), P.max(0)
    pos = [float(v) for v in a.pos.split(",")] if a.pos else list(0.5 * (lo + hi))
    labels = np.unique(lab)
    cmap = plt.get_cmap("tab20" if labels.max() > 10 else "tab10")
    color = lambda l: cmap(int(l) % cmap.N)  # noqa: E731
    names = a.names.split(",") if a.names else None
    axn = "xyz"

    panels = [(2, "axial"), (1, "coronal"), (0, "sagittal")]
    npan = len(panels) + (1 if a.zoom else 0)
    fig, axs = plt.subplots(1, npan, figsize=(5.2 * npan, 5.6))
    axial = None

    for ax, (axis, title) in zip(axs, panels):
        polys, cols, other = section(P, T, lab, axis, pos[axis])
        if axis == 2:
            axial = (polys, cols, other)
        pc = PolyCollection(
            polys, facecolors=[color(c) for c in cols], edgecolors="k", linewidths=a.lw
        )
        ax.add_collection(pc)
        ax.set_xlim(lo[other[0]], hi[other[0]])
        ax.set_ylim(lo[other[1]], hi[other[1]])
        ax.set_aspect("equal")
        ax.set_xlabel(axn[other[0]] + " (mm)")
        ax.set_ylabel(axn[other[1]] + " (mm)")
        ax.set_title(
            f"{title}: {axn[axis]} = {pos[axis]:.1f} mm ({len(polys)} elements cut)", fontsize=10
        )

    if a.zoom:
        f = [float(v) for v in a.zoom.split(",")]
        polys, cols, other = axial
        ax = axs[-1]
        pc = PolyCollection(
            polys, facecolors=[color(c) for c in cols], edgecolors="k", linewidths=4 * a.lw
        )
        ax.add_collection(pc)
        x0, x1 = (
            lo[other[0]] + f[0] * (hi - lo)[other[0]],
            lo[other[0]] + f[1] * (hi - lo)[other[0]],
        )
        y0, y1 = (
            lo[other[1]] + f[2] * (hi - lo)[other[1]],
            lo[other[1]] + f[3] * (hi - lo)[other[1]],
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title("axial, zoomed", fontsize=10)

    handles = [
        Patch(color=color(l), label=(names[l - 1] if names and 0 < l <= len(names) else str(l)))
        for l in labels
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 9), fontsize=8, frameon=False
    )
    fig.suptitle(f"{os.path.basename(a.mesh)}: {len(P)} nodes, {len(T)} tets", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(a.png, dpi=a.dpi)


if __name__ == "__main__":
    main()
