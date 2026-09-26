# SPDX-License-Identifier: GPL-3.0-or-later
"""trussnet -- GPU particle (truss) multi-label and gray-scale tetrahedral mesher.

    import trussnet
    out = trussnet.tetmesh(vol, size=3, gpu=True)
    node, elem, face = out["node"], out["elem"], out["face"]

See :func:`tetmesh`.
"""
from ._trussnet import tetmesh as _tetmesh
from ._trussnet import tetmesh_file as _tetmesh_file

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
        intensity; the interfaces are the sub-voxel iso-surfaces). A 4-D
        ``vol[x, y, z, c]`` is a tissue-probability map: the labels are the
        argmax of the classes; ``tpm_exterior=[channels]`` (0-based) are the
        exterior, else the exterior is 1 - sum(classes).
    faces : bool
        Also return the boundary / interface triangles (default True).
    affine : (4, 4) array, optional
        Voxel (0-based i, j, k) -> world matrix (e.g. ``nibabel`` ``img.affine``);
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
        1, 2, ...); K, grad, sigma, sigma_thin, thick, thin_floor, preserve;
        thresholds (list), gray_sigma; gpu (bool), gpuid (1-based device);
        reratio (alias q, default 2), opt, smooth, repair; iters, fscale,
        fsurf, dt, snap, nseed, jseed, corners, trap ('smooth' | 'voxel');
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
