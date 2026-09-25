"""Cut-away view of a trussnet JMesh -- the tets whose centroid is on one side of a
plane, drawn as a shaded 3-D view looking at the (jagged) cut, element edges
included -- plus the Joe-Liu quality and minimum-dihedral histograms.

Only the faces bounding the kept tets are drawn (their outer surface and the cut),
each coloured by its tet's label, projected orthographically and painted back to
front: fast and light enough for multi-million-tet meshes.

usage: python3 tools/tncut.py mesh.jmsh out.png [--axis y] [--pos P] [--keep ge|le]
       [--view elev,azim] [--names a,b,..] [--dpi N]
"""

import argparse
import base64
import json
import os
import resource
import zlib

import numpy as np

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


FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])


def outer_faces(T, n):
    """faces of the tet set T used once (its boundary), and their tet index"""
    F = T[:, FACES].reshape(-1, 3)  # 4 per tet
    owner = np.repeat(np.arange(len(T)), 4)
    s = np.sort(F, 1)
    key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    once = first[cnt == 1]
    return F[once], owner[once]


def quality(P, T):
    """Joe-Liu quality and the minimum dihedral angle (degrees) of every tet"""
    a, b, c, d = (P[T[:, k]] for k in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    e2 = sum(
        np.sum((P[T[:, i]] - P[T[:, j]]) ** 2, 1)
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    )
    jl = 12.0 * (3.0 * np.abs(vol)) ** (2.0 / 3.0) / e2
    # dihedral at edge (i, j): between the faces opposite the other two corners
    X = [a, b, c, d]
    nrm = []
    for f in FACES:  # face opposite corner k = 3 - index ... outward-agnostic normals
        p, q, r = (X[i] for i in f)
        v = np.cross(q - p, r - p)
        nrm.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300))
    dmin = np.full(len(T), 180.0)
    for i in range(4):
        for j in range(i + 1, 4):
            # the dihedral between faces i and j (opposite corners i and j)
            cosang = -np.einsum("ij,ij->i", nrm[i], nrm[j])
            # orient: normals of a tet's faces all point the same way (in / out) only
            # up to sign here; the interior dihedral is acos(-n_i . n_j) with consistent
            # outward normals -> fix the sign with the opposite corner
            oi = X[i] - X[FACES[i][0]]
            oj = X[j] - X[FACES[j][0]]
            si = np.sign(np.einsum("ij,ij->i", nrm[i], oi))
            sj = np.sign(np.einsum("ij,ij->i", nrm[j], oj))
            cosang = cosang * si * sj
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            dmin = np.minimum(dmin, ang)
    return jl, dmin


def camera(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # towards the camera
    right = np.cross([0.0, 0.0, 1.0], c)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(c, right)
    return c, right, up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("png")
    ap.add_argument("--axis", default="y", choices="xyz")
    ap.add_argument(
        "--pos", type=float, default=None, help="plane position (default: the mesh centre)"
    )
    ap.add_argument("--keep", default="ge", choices=("ge", "le"), help="keep centroids >= / <= pos")
    ap.add_argument(
        "--view", default=None, help="elev,azim (default: facing the cut, slightly tilted)"
    )
    ap.add_argument("--names", default=None, help="label names, comma separated (label 1 first)")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    P, T, lab = load(a.mesh)
    ax_i = "xyz".index(a.axis)
    lo, hi = P.min(0), P.max(0)
    pos = a.pos if a.pos is not None else 0.5 * (lo[ax_i] + hi[ax_i])
    cen = P[T].mean(1)[:, ax_i]
    sel = cen >= pos if a.keep == "ge" else cen <= pos
    F, own = outer_faces(T[sel], len(P))
    flab = lab[sel][own]

    # camera on the removed side, looking at the cut, tilted by 25 deg
    if a.view:
        elev, azim = (float(v) for v in a.view.split(","))
    else:
        side = -1.0 if a.keep == "ge" else 1.0
        elev, azim = {
            "x": (20.0, 90 + 90 * side - 30),
            "y": (20.0, 90 * side - 30),
            "z": (65 * side, -60.0),
        }[a.axis]
    c, right, up = camera(elev, azim)
    V = P[F]  # (nf, 3, 3)
    xy = np.stack([V @ right, V @ up], -1)  # (nf, 3, 2)
    depth = (V @ c).mean(1)
    nrm = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    shade = 0.35 + 0.65 * np.abs(nrm @ c)
    order = np.argsort(depth)  # far first
    labels = np.unique(lab)
    cmap = plt.get_cmap("tab20" if labels.max() > 10 else "tab10")
    base = np.array([cmap(int(l) % cmap.N)[:3] for l in flab])
    col = np.clip(base * shade[:, None], 0, 1)

    jl, dmin = quality(P, T)

    fig = plt.figure(figsize=(17, 6))
    ax = fig.add_axes([0.01, 0.1, 0.47, 0.82])
    lw = 0.06 if len(F) < 600000 else 0.03
    ax.add_collection(
        PolyCollection(xy[order], facecolors=col[order], edgecolors=(0, 0, 0, 0.45), linewidths=lw)
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    op = ">=" if a.keep == "ge" else "<="
    ax.set_title(
        f"tets with centroid {a.axis} {op} {pos:.1f} mm: {int(sel.sum())} of {len(T)} "
        f"({len(F)} faces drawn)",
        fontsize=10,
    )
    names = a.names.split(",") if a.names else None
    handles = [
        Patch(
            color=cmap(int(l) % cmap.N),
            label=(names[l - 1] if names and 0 < l <= len(names) else str(l)),
        )
        for l in labels
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.01),
        ncol=min(len(handles), 9),
        fontsize=7,
        frameon=False,
    )

    h1 = fig.add_axes([0.54, 0.14, 0.2, 0.72])
    h1.hist(jl, bins=60, range=(0, 1), color="steelblue")
    h1.set_yscale("log")
    h1.set_xlabel("Joe-Liu quality")
    h1.set_ylabel("tets")
    h1.set_title(
        f"Joe-Liu: min {jl.min():.3f}  P5 {np.percentile(jl, 5):.3f}  median {np.median(jl):.3f}",
        fontsize=9,
    )

    h2 = fig.add_axes([0.78, 0.14, 0.2, 0.72])
    h2.hist(dmin, bins=60, range=(0, 70.53), color="darkorange")
    h2.set_yscale("log")
    h2.set_xlabel("minimum dihedral angle (deg)")
    h2.set_title(
        f"min dihedral: min {dmin.min():.2f}  <5: {int((dmin < 5).sum())}  <10: {int((dmin < 10).sum())}  "
        f"median {np.median(dmin):.1f}",
        fontsize=9,
    )
    fig.suptitle(f"{os.path.basename(a.mesh)}: {len(P)} nodes, {len(T)} tets", fontsize=11)
    fig.savefig(a.png, dpi=a.dpi)


if __name__ == "__main__":
    main()
