"""Plot a trussnet JMesh: exterior surface, a cut-away (tets with centroid on the
far side of a plane, showing the interior elements), and the Joe-Liu histogram.

usage: python3 tools/tnplot.py mesh.jmsh out.png [--cut axis=value]
"""
import argparse, base64, json, zlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def jarray(o):
    raw = zlib.decompress(base64.b64decode(o['_ArrayZipData_']))
    return np.frombuffer(raw, dtype=o['_ArrayType_']).reshape(o['_ArraySize_'])


def load(path):
    d = json.load(open(path))
    P = jarray(d['MeshNode']).astype(float)
    E = jarray(d['MeshElem'])
    T = E[:, :4].astype(np.int64)
    if T.min() == 1:
        T -= 1
    return P, T, E[:, 4]


def boundary_faces(T, lab):
    """faces owned by exactly one tet, or shared by tets of different labels"""
    f = np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]])
    fl = np.tile(lab, 4)
    key = np.sort(f, axis=1)
    o = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    ks = key[o]
    same = np.all(ks[1:] == ks[:-1], axis=1)   # ks[i] == ks[i+1]
    keep = np.ones(len(f), bool)
    # a shared face appears twice in a row: keep neither if the labels agree
    pair = np.where(same)[0]
    agree = fl[o[pair]] == fl[o[pair + 1]]
    keep[o[pair[agree]]] = False
    keep[o[pair[agree] + 1]] = False
    keep[o[pair[~agree] + 1]] = False   # an interface face once
    return f[keep], fl[keep]


def joe_liu(P, T):
    a, b, c, d = (P[T[:, k]] for k in range(4))
    vol = np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6.0
    e2 = sum(np.sum((P[T[:, i]] - P[T[:, j]]) ** 2, axis=1)
             for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    return 12.0 * (3.0 * np.abs(vol)) ** (2.0 / 3.0) / e2


def draw(ax, P, F, fl, cmap, title, view):
    col = [cmap(int(l) % 10) for l in fl]
    lw = 0.08 if len(F) < 60000 else 0.0
    ax.add_collection3d(Poly3DCollection(P[F], facecolors=col, edgecolors='k', linewidths=lw))
    lo, hi = P.min(0), P.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo), zoom=1.1)   # true proportions, filling the panel
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh'); ap.add_argument('png')
    ap.add_argument('--cut', default='y', help='axis[=value] (default: mid-plane); tets with centroid below are removed (the cut faces the camera)')
    ap.add_argument('--view', default='22,-60', help='elevation,azimuth (degrees)')
    a = ap.parse_args()
    view = tuple(float(v) for v in a.view.split(','))
    P, T, lab = load(a.mesh)
    ax_i = 'xyz'.index(a.cut[0])
    cval = float(a.cut[2:]) if '=' in a.cut else 0.5 * (P[:, ax_i].min() + P[:, ax_i].max())
    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(15, 5.2))
    F, fl = boundary_faces(T, lab)
    draw(fig.add_subplot(131, projection='3d'), P, F, fl, cmap,
         f'{len(P)} nodes, {len(T)} tets, {len(set(lab.tolist()))} labels', view)

    cen = P[T].mean(axis=1)
    sel = cen[:, ax_i] >= cval
    Fc, flc = boundary_faces(T[sel], lab[sel])
    draw(fig.add_subplot(132, projection='3d'), P, Fc, flc, cmap, f'cut-away ({a.cut[0]} < {cval:.3g} removed)', view)

    q = joe_liu(P, T)
    ax = fig.add_subplot(133)
    ax.hist(q, bins=50, range=(0, 1), color='steelblue')
    ax.set_xlabel('Joe-Liu quality'); ax.set_ylabel('tets')
    ax.set_title(f'min {q.min():.3f}  P5 {np.percentile(q, 5):.3f}  median {np.median(q):.3f}', fontsize=10)
    fig.tight_layout()
    fig.savefig(a.png, dpi=130)


if __name__ == '__main__':
    main()
