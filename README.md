# trussnet

GPU particle (truss / moving-mesh) tetrahedral mesh generator for multi-label
volumes. It extends the DistMesh force-equilibrium idea (Persson & Strang 2004),
applied to binary volumes by Fang (SPIE 2006) and to multi-label volumes by
R. Walton (MS thesis, 2026), with everything mapped to voxel- and vertex-
parallel kernels.

Pipeline (label 0 = exterior, never meshed):
1. brick-sparse Gaussian-smoothed label indicators; curvature -> sizing field,
   gradient-limited, 256 log grades;
2. graded HCP seeding (one contiguous lattice per seeding level);
3. multi-level spatial hash, Verlet K-nearest truss (no Delaunay while moving);
4. DistMesh spring forces; nodes that cross an interface are trapped on it and
   glide (smooth sub-voxel interface or exact voxel faces), junction curves and
   corners constrained;
5. exact Delaunay of the relaxed nodes, tet labels from the nodes' label sets,
   exterior sculpting, restricted-Delaunay repair of crossing tets.

Status: CPU (OpenMP) reference of every stage; OpenCL kernels in progress.

    make
    ./build/trussnet --shape gyroid --dim 96 -o gyroid.bmsh
    ./build/trussnet -i labels.nii.gz --size 2 --gpu -o mesh.bmsh
    ./build/trussnet -i intensity.nii.gz --thresholds 2,2.5,3 --size 2 -o mesh.bmsh

## MATLAB / Octave and Python

Both bindings call the same pipeline as the command line (`src/tn_pipeline.h`),
take the same option names (case-insensitive, `_` ignored) and return
`node` (N x 3), `elem` (M x 5: 1-based `v1..v4`, label) and `face` (P x 5:
1-based `v1 v2 v3`, inner label, outer label; outer = 0 on the exterior surface,
normals from inner to outer). A gray-scale volume is meshed at its iso-surfaces
when `thresholds` is given.

    make bindings      # Python module + MATLAB / Octave MEX (build-bind/)
    make test          # their unit tests (ctest)

MATLAB / Octave (`addpath matlab`):

    [node, elem, face, info] = trussnet(vol, 'size', 3, 'gpu', 1);
    [node, elem, face] = trussnet(img, 'thresholds', [2 2.5 3], 'size', 2);
    plotmesh(node, face(:, 1:4));   % iso2mesh

Python (`pip install ./pytrussnet`, or `PYTHONPATH=pytrussnet` in-tree):

    import trussnet
    out = trussnet.tetmesh(vol, size=3, gpu=True, lsize={2: 1.5})
    out = trussnet.tetmesh_file("head.nii.gz", size=3)      # world coordinates (nibabel)
    node, elem, face = out["node"], out["elem"], out["face"]

Node coordinates: in MATLAB, index space scaled by `voxelsize` (voxel `(i,j,k)`
at `[i j k] .* voxelsize`); in Python, `[i, j, k] * voxelsize` (0-based); with
`affine` (4 x 4, 0-based voxel -> world, e.g. a NIfTI header's) world coordinates
in both. `make pretty` formats the sources (astyle, black, mh_style).

License: GPL-3.0-or-later. Vendored: siamize volume I/O (Apache-2.0), zmat /
miniz, nlohmann/json (MIT), the exact Delaunay/CDT of Diazzi et al. (third_party/cdt).
