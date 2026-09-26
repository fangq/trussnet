# SPDX-License-Identifier: GPL-3.0-or-later
"""trussnet -- GPU particle (truss) multi-label and gray-scale tetrahedral mesher.

    import trussnet
    out = trussnet.tetmesh(vol, size=3, gpu=True)
    node, elem, face = out["node"], out["elem"], out["face"]

See :func:`tetmesh`.
"""
from ._trussnet import tetmesh as _tetmesh
from ._trussnet import tetmesh_file as _tetmesh_file
from ._trussnet import trimesh as _trimesh

__version__ = "0.5.0"
__all__ = ["tetmesh", "tetmesh_file", "trimesh"]


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
