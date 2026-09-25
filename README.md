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
    ./build/trussnet -i labels.nii.gz --size 2 -o mesh.bmsh

License: GPL-3.0-or-later. Vendored: siamize volume I/O (Apache-2.0), zmat /
miniz, nlohmann/json (MIT), the exact Delaunay/CDT of Diazzi et al. (third_party/cdt).
