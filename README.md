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
    ./build/trussnet -i tpm.bnii --gpu -o mesh.bmsh          # 4-D tissue-probability map

## Tissue-probability maps (TPM)

A 4-D input (`.jnii`, `.bnii`, `.nii`, `.nii.gz`; channels last) is read as
per-voxel class probabilities, e.g. SPM's 6 classes or siamize's 18. Channel
names come from a JNIfTI `LabelTable`; channels named background / air are the
exterior (label 0), every other channel is its own label (`--tpm-exterior`,
`--tpm-map L0,L1,..` to merge, `--tpm-spm6` for siamize -> SPM6). Without an
exterior channel the exterior is `1 - sum(classes)`. The labels are the argmax;
exterior pockets enclosed by tissue are filled (`--tpm-holes` keeps them), and
no exterior probability is left deeper than 2 voxels inside the tissue.

By default the argmax labels are meshed like a label volume. `--tpm-fields`
places the interfaces at the probabilities' own crossings `p_a = p_b` instead;
with `--sigma 0` (unsmoothed) that is the most accurate for a genuinely smooth
TPM (a synthetic r = 7 ball: -3% vs -7% volume), while the default is more robust
on real TPMs, whose boundaries are often step-like (a network's softmax, a
binary atlas head surface). Results (Titan V, `--gpu`):

| TPM | tets | bad faces / outside edges | min dihedral | volumes | time |
|---|---|---|---|---|---|
| ANTS 40-44 y atlas, 5 classes (brain2mesh sample) | 956k | 1340 / 0 | 0.63 deg | within 1.8% | 7.9 s |
| siamize SPM6 (160x192x192) | 6.8M | 2317 / 9 | 1.32 deg | CSF -6%, others within 2% | 40 s |
| siamize 18 classes (17 labels) | 7.3M | 5245 / 2 | 1.53 deg | nuclei within 6.5% | 51 s |

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
