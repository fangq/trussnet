# SPDX-License-Identifier: GPL-3.0-or-later
"""trussnet -- GPU particle (truss) multi-label and gray-scale tetrahedral mesher.

    import trussnet
    out = trussnet.tetmesh(vol, size=3, gpu=True)
    node, elem, face = out["node"], out["elem"], out["face"]

See :func:`tetmesh`.
"""
from ._trussnet import tetmesh as _tetmesh

__version__ = "0.1.0"
__all__ = ["tetmesh", "tetmesh_file"]


def tetmesh(vol, *, faces=True, affine=None, voxelsize=None, **opts):
    """Tetrahedral mesh of a 3-D label or gray-scale volume.

    Parameters
    ----------
    vol : ndarray, 3-D, indexed ``vol[x, y, z]`` (as nibabel's ``get_fdata()``)
        Integer labels (0 = exterior, never meshed; 1..N tissues, meshed with
        conforming shared interfaces), or a gray-scale intensity when
        ``thresholds`` is given (a voxel's label = the number of thresholds <= its
        intensity; the interfaces are the sub-voxel iso-surfaces).
    faces : bool
        Also return the boundary / interface triangles (default True).
    affine : (4, 4) array, optional
        Voxel (0-based i, j, k) -> world matrix (e.g. ``nibabel`` ``img.affine``);
        the voxel size (element sizes are in mm) is taken from its columns.
    voxelsize : float or 3 floats, optional
        Voxel size in mm (default 1) when no affine is given; the nodes are then
        ``[i, j, k] * voxelsize``.
    **opts
        size, hmin, hmax (mm); lsize ({label: size} or a sequence for labels
        1, 2, ...); K, grad, sigma, sigma_thin, thick, thin_floor, preserve;
        thresholds (list), gray_sigma; gpu (bool), gpuid (1-based device);
        reratio (alias q, default 2), opt, smooth, repair; iters, fscale,
        fsurf, dt, snap, nseed, jseed, corners, trap ('smooth' | 'voxel');
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


def tetmesh_file(path, **kw):
    """:func:`tetmesh` of a NIfTI file (needs nibabel), in its world coordinates."""
    import nibabel as nib
    import numpy as np

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if "thresholds" not in kw and not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    return tetmesh(data, affine=img.affine, **kw)
