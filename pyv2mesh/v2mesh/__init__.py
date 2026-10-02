# SPDX-License-Identifier: GPL-3.0-or-later
"""v2mesh -- GPU particle (truss) multi-label and gray-scale tetrahedral mesher.

    import v2mesh
    out = v2mesh.tetmesh(vol, size=3, gpu=True)
    node, elem, face = out["node"], out["elem"], out["face"]

See :func:`tetmesh`.
"""
from ._v2mesh import tetmesh as _tetmesh
from ._v2mesh import tetmesh_file as _tetmesh_file
from ._v2mesh import trimesh as _trimesh
from ._v2mesh import check as _check
from ._v2mesh import tessellate as _tessellate
from ._v2mesh import optimize as _optimize
from ._v2mesh import cdt as _cdt
from ._v2mesh import remesh as _remesh

__version__ = "0.5.0"
__all__ = [
    "tetmesh",
    "tetmesh_file",
    "shapes",
    "trimesh",
    "surface",
    "points",
    "tessellate",
    "optimize",
    "cdt",
    "remesh",
    "repair",
    "check",
]


def _surfaces_only(out):
    """keep ``node`` / ``face`` / ``info``, the nodes renumbered to the faces' own"""
    import numpy as np

    face = out["face"].copy()
    used, inv = np.unique(face[:, :3].ravel() - 1, return_inverse=True)
    face[:, :3] = inv.reshape(-1, 3) + 1
    return {"node": out["node"][used], "face": face, "info": out["info"]}


def tetmesh(vol, *, faces=True, affine=None, voxelsize=None, **opts):
    """Tetrahedral mesh of a 3-D label or gray-scale volume.

    Parameters
    ----------
    vol : ndarray, 3-D, indexed ``vol[x, y, z]`` (x the first axis)
        Integer labels (0 = exterior, never meshed; 1..N tissues, meshed with
        conforming shared interfaces), or a gray-scale intensity when
        ``thresholds`` is given (a voxel's label = the number of thresholds <= its
        intensity; the interfaces are the sub-voxel iso-surfaces). A 4-D
        ``vol[x, y, z, c]`` is a tissue-probability map: the labels are the
        argmax of the classes; ``tpm_exterior=[channels]`` (0-based) are the
        exterior, else the exterior is 1 - sum(classes).
    faces : bool
        Also return the boundary / interface triangles (default True).
    affine : (4, 4) array, optional
        Voxel (0-based i, j, k) -> world matrix (e.g. a NIfTI header's sform);
        the voxel size (element sizes are in mm) is taken from its columns.
    voxelsize : float or 3 floats, optional
        Voxel size in mm (default 1) when no affine is given; the nodes are then
        ``[i, j, k] * voxelsize``.
    sizing : array, optional (keyword)
        A user sizing (mm): an array with ``vol``'s spatial shape (a sizing
        field; 0 = the automatic size at that voxel), or a sequence with one size
        per label (N values for labels 1..N, or N+1 from label 0), per threshold
        level (gray-scale) or per channel (TPM); 0 = the default size.
    **opts
        size, hmin, hmax (mm); lsize ({label: size} or a sequence for labels
        1, 2, ...); isize, the size at the interfaces only (a number for every
        interface, {label: h} for every interface of a label, 0 = the outer
        surface, {(a, b): h} for one interface, or a string "h,L:h,A:B:h"),
        e.g. size=6, isize={0: 2, (3, 4): 1.5}; thin (seed thinning, e.g. 0.7); tpm_thresh (per-label TPM threshold, default
        0.5 = the argmax: a number, {label: t}, a sequence for labels 1.., or "T,L:T"); K, grad, sigma, sigma_thin, thick, thin_floor, preserve;
        thresholds (list), gray_sigma; gpu (bool), gpuid (1-based device);
        reratio (alias q, default 2), maxvol (cdt / optimize: the largest tet volume, TetGen -a), surf_smooth (volume-preserving
        surface smoothing passes) with surf_smooth_method ('laplacianhc', 'lowpass', 'laplacian') and
        surf_smooth_alpha / surf_smooth_beta, opt, smooth, repair; iters, fscale,
        fsurf, dt, snap, nseed, jseed, corners, trap ('smooth' | 'voxel'),
        relax ('fire', the default, | 'jacobi'), fire_dtmax, dptol;
        TPM input: tpm_exterior, tpm_map (label per channel, 0 = exterior),
        tpm_spm6 (merge siamize's 18 classes), tpm_sigma, tpm_holes (keep the
        enclosed exterior pockets), tpm_fields (probability interfaces);
        verbose.

    Returns
    -------
    dict
        ``node`` (N, 3) float64; ``elem`` (M, 5) int32 ``[v1..v4, label]``,
        1-based; ``face`` (P, 5) int32 ``[v1, v2, v3, inner, outer]``, 1-based,
        outer = 0 on the exterior surface, normals from inner to outer;
        ``info``: counts, conformity (bad_faces / bad_edges / spanning, 0 =
        conforming), quality and timings.
    """
    return _tetmesh(vol, faces=faces, affine=affine, voxelsize=voxelsize, **opts)


def tetmesh_file(path, *, faces=True, **opts):
    """:func:`tetmesh` of a volume file, in its world coordinates.

    ``.nii`` / ``.nii.gz`` / ``.jnii`` / ``.bnii``: a 3-D label volume, a gray-scale
    volume (with ``thresholds``), or a 4-D tissue-probability map (channel names
    from a JNIfTI LabelTable; background / air channels are the exterior).
    """
    return _tetmesh_file(str(path), faces=faces, **opts)


def shapes(src, *, faces=True, **opts):
    """:func:`tetmesh` of shape constructs: MCX ``Shapes`` lists (``Grid``, ``Box``,
    ``Sphere``, ``Cylinder``, ...) or JMesh ``Shape*`` primitives with ``CSG*``.

    ``src``: a dict / list (the parsed JSON), JSON text, or a ``.json`` file. The
    objects overwrite one another in order, all cut to the first (``shape_clip=
    False``: not); each shape is an exact signed distance function that the mesher
    evaluates directly, sharp edges and corners pinned. ``raster_voxel``: the
    spacing of the raster the sizing is computed on (default size / 3).
    ``overlap``: who owns a volume two objects claim -- ``"overwrite"`` (default:
    the later one, as MCX), ``"nest"`` (the smaller), ``"max"`` / ``"min"``
    (label), ``"order:L1,L2,.."``, ``"split"`` (halfway), ``"union"`` (one region)
    or ``"cells"`` (each overlap a region of its own, every surface kept). Returns
    what :func:`tetmesh` does, in the shapes' coordinates.
    """
    import json

    if isinstance(src, (dict, list)):
        src = json.dumps(src)
    return _tetmesh_file(str(src), faces=faces, **opts)


def trimesh(img, *, faces=True, affine=None, pixelsize=None, **opts):
    """Triangle mesh of a 2-D label or gray-scale image (the same moving-particle method).

    Parameters
    ----------
    img : ndarray, 2-D, indexed ``img[x, y]``
        Integer labels (0 = exterior; 1..N meshed with conforming shared interface
        curves), or a gray-scale intensity when ``thresholds`` is given (a
        pixel's label = the number of thresholds <= its intensity; the interfaces
        are the sub-pixel iso-lines).
    faces : bool
        Also return the boundary / interface edges.
    affine : (2, 3) or (3, 3) array, optional
        Pixel (0-based i, j) -> world matrix; the pixel size from its columns.
    pixelsize : float or 2 floats, optional
        Pixel size (default 1) when no affine is given; nodes = [i, j] * pixelsize.
    **opts
        size, hmin, hmax (mm), lsize ({label: size} or a sequence for labels 1..),
        isize (interface sizes, as tetmesh),
        sizing (a per-pixel field with img's shape, 0 = automatic; or one size per
        label / threshold level), K, grad, sigma, thresholds, gray_sigma, nseed,
        iters, fscale, fsurf, dt, snap, repair, smooth, verbose.

    Returns
    -------
    dict
        ``node`` (N, 2) float64; ``elem`` (M, 4) int32 ``[v1, v2, v3, label]``,
        1-based, counter-clockwise; ``face`` (P, 4) int32 ``[v1, v2, inner, outer]``,
        1-based: the boundary (outer = 0) and each interface edge once, the inner
        region on the left of v1 -> v2; ``info``: counts, conformity
        (bad_edges / spanning, 0 = conforming), quality (min_angle, q_* with
        q = 4 sqrt(3) A / sum(l^2), 1 = equilateral), per-label areas, timings.
    """
    return _trimesh(img, faces=faces, affine=affine, pixelsize=pixelsize, **opts)


def surface(vol, *, affine=None, voxelsize=None, **opts):
    """The region and exterior surfaces of a volume (:func:`tetmesh`, faces only).

    Returns ``node`` and ``face`` (P, 5) ``[v1, v2, v3, inner, outer]``, 1-based
    (outer = 0: the exterior), closed and conforming, and ``info``. A string or
    path ``vol`` is read as a file (as :func:`tetmesh_file`).
    """
    opts.setdefault("surface_only", True)  # only the surface nodes tessellated (exact_tess: all)
    if opts.pop("exact_tess", False):
        opts["surface_only"] = False
    if isinstance(vol, str) or hasattr(vol, "__fspath__"):
        return _surfaces_only(_tetmesh_file(str(vol), faces=True, **opts))
    return _surfaces_only(_tetmesh(vol, faces=True, affine=affine, voxelsize=voxelsize, **opts))


def points(vol, *, affine=None, voxelsize=None, **opts):
    """The mesher's relaxed nodes, before tessellation.

    Returns ``node`` (N, 3), ``label`` (N,) (own label), ``type`` (N,) (0
    interior, 1 interface, 2 junction, 3 corner), ``partner`` (N, 3) (the other
    labels, -1 = none) and ``info``. :func:`tessellate` meshes them.
    """
    if isinstance(vol, str) or hasattr(vol, "__fspath__"):
        out = _tetmesh_file(str(vol), faces=False, stop_after_relax=True, **opts)
    else:
        out = _tetmesh(
            vol, faces=False, affine=affine, voxelsize=voxelsize, stop_after_relax=True, **opts
        )
    out.pop("elem", None)
    return out


def tessellate(node, label=None, *, gpu=None):
    """The Delaunay tets of a point cloud (their convex hull).

    ``node`` (N, 3); ``label`` (N,) optional: each tet takes the most frequent
    label of its nodes (else 1). ``gpu``: an OpenCL device for the Delaunay
    (True = the first GPU). Returns ``node``, ``elem`` (M, 5), 1-based, ``info``.
    """
    dev = -2 if gpu is None or gpu is False else (-1 if gpu is True else int(gpu))
    return _tessellate(node, label, dev)


def optimize(node, elem, *, opt_rounds=3, **opts):
    """The mesher's optimiser alone on a labelled tet mesh.

    ``elem`` (M, 4 or 5), 1-based (label in the 5th column, else 1). Flips, kite
    removal, collapses, Steiner points and guarded smoothing, after refinement to
    the radius-edge bound ``q`` (default 2; 0 = off): circumcentres inserted only
    inside a region, never encroaching an interface. The region interfaces and
    the boundary are kept.
    Returns ``node``, ``elem``, ``info`` (quality and the operations applied).
    """
    return _optimize(node, elem, opt_rounds=opt_rounds, **opts)


def cdt(node, face, *, fill=None, faces=True, opt_rounds=3, **opts):
    """Labelled tets of closed surfaces, the surfaces kept exactly (constrained Delaunay).

    ``face`` (P, 3), (P, 4) ``[.., label]`` or (P, 5) ``[.., inner, outer]``,
    1-based; closed and not self-intersecting (see :func:`check`, :func:`repair`).
    Regions: from the inner / outer labels; else from the cells the surfaces
    enclose, exactly: a face labelled ``l`` bounds region ``l`` (the other side is
    what surrounds it; a face given twice, labels ``a`` and ``b``, lies between
    them); unlabelled, each cell is a region (outward shells: the ones they wind
    around; ``auto_labels="cell"``, outermost then largest first, or
    ``"depth"``). ``fill``:
    the spacing of interior points (default: ``size``, else 1.5 x the mean edge;
    0 = none); then ``q`` refinement and the optimiser (as :func:`optimize`)
    unless ``opt=False``. Returns ``node``,
    ``elem``, ``face`` (the region surfaces) and ``info``.
    """
    return _cdt(
        node,
        face,
        fill=-1.0 if fill is None else float(fill),
        faces=faces,
        opt_rounds=opt_rounds,
        **opts,
    )


def remesh(node, face, *, raster_voxel=None, faces=True, **opts):
    """Labelled tets of the regions enclosed by closed surfaces, which may self-intersect.

    The surfaces (regions as :func:`cdt`, exact where they do not cross; the
    orientation repaired) are rasterized into per-region soft fields
    (``raster_voxel``, default: the smaller of size / 3 and half the mean edge),
    which the whole mesher then meshes (every :func:`tetmesh` option applies).
    Where regions cross, ``overlap`` says who owns a volume two claim: ``"nest"``
    (default: the smaller region), ``"split"`` (halfway), ``"max"`` / ``"min"``
    (label), ``"order:L1,L2,.."``, ``"union"`` (one region) or ``"cells"`` (each
    overlap a region of its own).
    Returns what :func:`tetmesh` does, in the surfaces' coordinates.
    """
    return _remesh(
        node,
        face,
        raster_voxel=0.0 if raster_voxel is None else float(raster_voxel),
        faces=faces,
        **opts,
    )


def repair(node, face, **opts):
    """:func:`remesh`, returning only the region surfaces: closed, no self-intersections."""
    opts.setdefault("surface_only", True)
    if opts.pop("exact_tess", False):
        opts["surface_only"] = False
    return _surfaces_only(
        _remesh(
            node, face, raster_voxel=float(opts.pop("raster_voxel", 0) or 0), faces=True, **opts
        )
    )


def check(node, elem=None, face=None):
    """A report on a tet mesh and / or a surface (1-based ``elem`` / ``face``).

    Returns a dict: quality (min_dihedral, joe_liu_*), ``inverted`` tets, the
    ``open_edges`` / ``junction_edges`` / ``region_open_edges`` of the surfaces
    (a tet mesh: its region surfaces), ``self_intersections``, and ``ok``.
    """
    return _check(node, elem, face)
