# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
"""The surface mesh of each label -- the boundary of the region of the tets with
that label (its outer surface, and any inner one) -- one shaded 3-D panel per
label, element edges drawn, faces turned away from the camera culled.

--nested: layered models (a head: scalp, skull, CSF, GM, WM): each label is joined
with all the layers inside it (--order, outermost first; default the label order),
and the surfaces around enclosed cavities (the ventricles) are dropped, so that its
panel shows only its exterior surface.

--zoom W[,fx,fy]: a second row, each panel zoomed to a W x W mm window centred at
(fx, fy) of the view (fractions, default 0.5,0.5), element edges drawn heavier --
dense surfaces (triangles smaller than a pixel in the full view) need it.

usage: python3 tools/tnsurf.py mesh.jmsh out.png [--labels 1,2,..] [--names a,b,..]
       [--nested [--order 1,2,..]] [--zoom W[,fx,fy]] [--view elev,azim] [--dpi N]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tncut import camera, load, outer_faces  # noqa: E402  (also sets the memory cap)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])


def outward(P, T, F, own):
    """flip each boundary face so that its normal points away from its own tet"""
    V = P[F]
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    inside = P[T[own]].mean(1) - V[:, 0]  # towards the tet's centroid
    flip = np.einsum("ij,ij->i", n, inside) > 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def drop_cavities(P, F):
    """keep the pieces of an outward-oriented surface that bound the region from
    outside: a piece around a cavity (e.g. the ventricles inside the brain) encloses
    a negative volume and is dropped"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf = len(F)
    rows = np.repeat(np.arange(nf), 3)
    g = coo_matrix((np.ones(3 * nf), (rows, F.ravel())), shape=(nf, len(P))).tocsr()
    _, comp = connected_components(g @ g.T, directed=False)  # faces sharing a node
    V = P[F]
    vol = np.einsum("ij,ij->i", V[:, 0], np.cross(V[:, 1], V[:, 2])) / 6.0
    keep = np.bincount(comp, weights=vol) > 0
    return F[keep[comp]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("png")
    ap.add_argument("--labels", default=None, help="labels to draw (default: all)")
    ap.add_argument("--names", default=None, help="label names, comma separated (label 1 first)")
    ap.add_argument(
        "--nested", action="store_true", help="join each label with the layers inside it"
    )
    ap.add_argument(
        "--order", default=None, help="--nested: the labels outermost first (default ascending)"
    )
    ap.add_argument("--zoom", default=None, help="W[,fx,fy]: a zoomed row, W mm window")
    ap.add_argument("--view", default="10,180", help="elev,azim (default 10,180: from -x)")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    P, T, lab = load(a.mesh)
    labels = [int(v) for v in a.labels.split(",")] if a.labels else sorted(set(lab.tolist()))
    names = a.names.split(",") if a.names else None
    layers = [int(v) for v in a.order.split(",")] if a.order else sorted(set(lab.tolist()))
    cmap = plt.get_cmap("tab20" if max(labels) > 10 else "tab10")
    elev, azim = (float(v) for v in a.view.split(","))
    c, right, up = camera(elev, azim)
    lo, hi = P.min(0), P.max(0)
    box = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    bx, by = box @ right, box @ up

    N = len(P)
    fkey = lambda F: (lambda q: (q[:, 0] * N + q[:, 1]) * N + q[:, 2])(np.sort(F, 1))  # noqa: E731
    skin = fkey(outer_faces(T, N)[0])  # the whole mesh's outer surface (label 0 beyond)

    def median_edge(F):
        if not len(F):
            return float("nan")
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
        return float(np.median(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)))

    zw = None
    if a.zoom:
        z = [float(v) for v in a.zoom.split(",")]
        fx, fy = (z[1], z[2]) if len(z) >= 3 else (0.5, 0.5)
        zx, zy = bx.min() + fx * np.ptp(bx), by.min() + fy * np.ptp(by)
        zw = (zx - z[0] / 2, zx + z[0] / 2, zy - z[0] / 2, zy + z[0] / 2)
    nrow = 2 if zw else 1

    def tri_quality(F):
        """q = 4 sqrt(3) area / sum(edge^2) (1 = equilateral) and the smallest angles"""
        V = P[F]
        e = [V[:, 1] - V[:, 0], V[:, 2] - V[:, 1], V[:, 0] - V[:, 2]]
        l2 = np.stack([np.einsum("ij,ij->i", x, x) for x in e], 1)
        area = 0.5 * np.linalg.norm(np.cross(e[0], -e[2]), axis=1)
        q = 4 * np.sqrt(3) * area / np.maximum(l2.sum(1), 1e-300)
        L = np.sqrt(l2)  # the angle opposite each edge (law of cosines)
        ang = np.degrees(
            np.arccos(
                np.clip(
                    (np.roll(l2, 1, 1) + np.roll(l2, 2, 1) - l2)
                    / np.maximum(2 * np.roll(L, 1, 1) * np.roll(L, 2, 1), 1e-300),
                    -1,
                    1,
                )
            )
        ).min(1)
        return q, ang

    fig, axs = plt.subplots(
        nrow, len(labels), figsize=(5.2 * len(labels), 5.8 * nrow), squeeze=False
    )
    for k, l in enumerate(labels):
        ax = axs[0][k]
        inner = layers[layers.index(l) :] if a.nested and l in layers else [l]
        sub = np.where(np.isin(lab, inner))[0]
        F, own = outer_faces(T[sub], len(P))
        F = outward(P, T[sub], F, own)
        if a.nested:  # only the exterior: no surfaces around enclosed cavities
            F = drop_cavities(P, F)
        V = P[F]
        n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        front = n @ c > 0
        V, n = V[front], n[front]
        order = np.argsort((V @ c).mean(1))  # far first
        shade = 0.3 + 0.7 * (n @ c)
        base = np.array(cmap(l % cmap.N)[:3])
        col = np.clip(base[None, :] * shade[:, None], 0, 1)
        xy = np.stack([V @ right, V @ up], -1)
        lw = 0.08 if len(V) < 200000 else 0.04
        ax.add_collection(
            PolyCollection(
                xy[order], facecolors=col[order], edgecolors=(0, 0, 0, 0.5), linewidths=lw
            )
        )
        ax.set_xlim(bx.min(), bx.max())
        ax.set_ylim(by.min(), by.max())
        ax.set_aspect("equal")
        ax.set_axis_off()
        if zw:
            x0, x1, y0, y1 = zw
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
            q = xy[order]
            inw = (q[..., 0].max(1) > x0) & (q[..., 0].min(1) < x1)
            inw &= (q[..., 1].max(1) > y0) & (q[..., 1].min(1) < y1)
            az = axs[1][k]
            az.add_collection(
                PolyCollection(
                    q[inw], facecolors=col[order][inw], edgecolors=(0, 0, 0, 0.85), linewidths=0.3
                )
            )
            az.set_xlim(x0, x1)
            az.set_ylim(y0, y1)
            az.set_aspect("equal")
            az.set_axis_off()
            az.set_title(f"zoomed: {x1 - x0:.0f} x {y1 - y0:.0f} mm window", fontsize=10)
        out = np.isin(fkey(F), skin)  # faces on the outer surface vs on other tissues
        if a.nested and len(inner) > 1:
            name_extra = " + inner layers"
        else:
            name_extra = ""
        name = names[l - 1] if names and 0 < l <= len(names) else f"label {l}"
        sizes = []
        if out.any():
            sizes.append(f"outer {median_edge(F[out]):.2f}")
        if (~out).any():
            sizes.append(f"interfaces {median_edge(F[~out]):.2f}")
        q, amin = tri_quality(F)
        ax.set_title(
            f"{name}{name_extra}: {len(F)} triangles\nmedian edge (mm): "
            + ", ".join(sizes)
            + f"\nq median {np.median(q):.3f}, min angle < 20 deg: {100 * np.mean(amin < 20):.2f}%",
            fontsize=10,
        )
    fig.suptitle(f"{os.path.basename(a.mesh)}: {len(P)} nodes, {len(T)} tets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.png, dpi=a.dpi)


if __name__ == "__main__":
    main()
