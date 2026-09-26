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

## 2-D images

A 2-D image (a MATLAB 2-D array, `trussnet.trimesh` in Python, or a single-slice
volume on the command line) is meshed into triangles by the same method in its own
unit (`src/tn_2d.cpp`, CPU/OpenMP): smoothed label fields (sigma 0.5) or gray-scale
memberships with `thresholds`, curvature sizing with gradient limiting, graded
hexagonal seeding with fixed nodes at the junctions of >= 3 labels, spring
relaxation with the nodes trapped on and gliding along the interface curves
(interface density control), an exact Delaunay (Bowyer-Watson on the vendored
orient2d / incircle), label-set triangle labels, conformity repair and cap flips.
Options and `sizing` as in 3-D.

    [node, elem, face, info] = trussnet(img, 'size', 3);                    % labels
    [node, elem, face] = trussnet(img, 'thresholds', [0.4 0.8], 'size', 3);  % gray-scale
    out = trussnet.trimesh(img, size=3, lsize={2: 1.5})

`elem` is M x 4 `[v1 v2 v3 label]` (counter-clockwise), `face` P x 4 `[v1 v2 inner
outer]` (boundary and interface edges, the inner region on the left). A Colin27
axial slice (183 x 219, 6 labels): 29.6k triangles in 0.4 s at size 1.5,
conforming, areas within 1.3% (thin CSF -7%); a 3-level gray-scale image: 3.5k
triangles in 0.07 s, min angle 18.7 deg, areas within 0.2%.

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
    out = trussnet.tetmesh_file("head.nii.gz", size=3)      # a volume file, in its world coordinates
    node, elem, face = out["node"], out["elem"], out["face"]

A user sizing is the `sizing` option: an array with the volume's (spatial) size
is a sizing field in mm (0 = the automatic size at that voxel; the gradient limit
still grades it), and a vector gives one size per label (N values for labels 1..N,
or N+1 from label 0), per threshold level of a gray-scale volume, or per channel
of a TPM (0 = default):

    [node, elem] = trussnet(vol, 'size', 3, 'sizing', 1.5 * (dist < 10));   % a field
    out = trussnet.tetmesh(tpm, sizing=[0, 2, 2.5, 4, 6, 0])                 # per channel

Node coordinates: in MATLAB, index space scaled by `voxelsize` (voxel `(i,j,k)`
at `[i j k] .* voxelsize`); in Python, `[i, j, k] * voxelsize` (0-based); with
`affine` (4 x 4, 0-based voxel -> world, e.g. a NIfTI header's) world coordinates
in both. `make pretty` formats the sources (astyle, black, mh_style).

## Tests and CI

    make check      # the standalone binary + its CLI tests (tests/run_cli_tests.sh)
    make test       # also the Python module and the MATLAB / Octave MEX, all their unit tests

GitHub Actions (`.github/workflows`): `ci.yml` builds the binary with its CLI
tests on Linux, macOS and Windows (MSYS2, static; artifacts uploaded), a CPU-only
build, the Python module, the Octave and MATLAB MEX with their unit tests, and the
`make pretty` formatting gate; `wheels.yml` builds and tests Python wheels
(cibuildwheel on Linux / macOS, MinGW on Windows) and uploads new ones to PyPI.
The runners have no GPU: `--gpu` falls back to the CPU there.

License: GPL-3.0-or-later. Vendored: siamize volume I/O (Apache-2.0), zmat /
miniz, nlohmann/json (MIT), the exact Delaunay/CDT of Diazzi et al. (third_party/cdt).
