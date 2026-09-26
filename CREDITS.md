# Credits, third-party code and licenses

trussnet is GPL-3.0-or-later ([LICENSE](LICENSE)). It carries, adapts, compiles in
or (in binary packages) bundles the works below; their license texts are in
[LICENSES/](LICENSES/), and each adopted source file keeps an attribution header.

## Vendored source (carried in this repository)

| Path | Origin | Authors | License |
|---|---|---|---|
| `third_party/cdt/` | [CDT](https://github.com/MarcoAttene/CDT): the exact Delaunay / constrained Delaunay tetrahedrization, with Attene's indirect geometric predicates (`numerics`, `implicit_point`, `indirect_predicates`, `hand_optimized_predicates`, `memPool`) | Lorenzo Diazzi, Daniele Panozzo, Amir Vaxman, Marco Attene (IMATI-GE / CNR) | LGPL-3.0-or-later (`third_party/cdt/lgpl.txt`). Its optional `USE_MAROTS_METHOD` path (derived from Célestin Marot's GPL `hxt_SeqDel`) is **off** in trussnet. Modified for C++11: the two `inline static thread_local` pools in `include/numerics.h` (the class member is defined in `src/delaunay.cpp` before C++17). |
| `third_party/nlohmann/json.hpp` | [nlohmann/json](https://github.com/nlohmann/json) | Niels Lohmann | MIT |
| `third_party/zmat/zmat.h` | [zmat](https://github.com/NeuroJSON/zmat), amalgamated with [miniz](https://github.com/richgel999/miniz) | Qianqian Fang; miniz: Rich Geldreich | zmat: GPL-3.0, this copy dual-licensed Apache-2.0 by its author; miniz: public domain (Unlicense), as marked in its block |
| `src/io/` (`nifti_io`, `jnifti_io`, `orient`, `siam.h`) | [siamize](https://github.com/NeuroJSON/siamize) volume I/O | Qianqian Fang | Apache-2.0 |

LGPL-3.0 and Apache-2.0 code may be combined into this GPL-3.0-or-later work; their
notices and license texts must travel with it (source and binaries).

## Derived / adapted code in trussnet's own sources

- `src/opencl/tn_del_body.cl` -- the device Delaunay's floating-point filters
  (`orient3d_filtered`, `inSphere_filtered`) are transcribed from Attene's
  indirect predicates (LGPL-3.0-or-later, above).
- `src/tn_gdel.cpp`, `src/opencl/tn_del_kernels.cl` -- original parallel
  Bowyer-Watson code, written against the CDT `TetMesh` data structure (corner
  encoding, ghost tets, symbolic perturbation) so its output loads into it; the
  deferred points are inserted by the CDT's own `insertExistingVertex`.
- `src/tn_2d.cpp` -- calls the CDT's exact `orient2d` / `incircle`.
- `src/tn_opt.cpp` -- the sliver-repair passes are ported from
  [gpu_brain2mesh](https://github.com/NeuroJSON/gpu_brain2mesh) (same author,
  GPL-3.0-or-later). Its radius-edge test `b2m_check_bad` descends, through
  gpu_brain2mesh, from **gQM3d**'s `checktet4split` (Zhenghai Chen, Tiow-Seng Tan,
  National University of Singapore; BSD-3-Clause, `LICENSES/BSD-3-Clause-gQM3d.txt`).
- `src/opencl/tn_cl_host.*`, `src/tn_jmesh.*`, `src/tn_volume.*`, `src/tn_omp.h`,
  `src/tn_tpm.*` -- adapted from gpu_brain2mesh (same author, GPL-3.0-or-later).
- `.github/` -- CI adapted from blit, gpu_brain2mesh and mmc (same author).

## Compiled in, linked, or bundled in binary packages

| Component | Where | License | Note |
|---|---|---|---|
| [pybind11](https://github.com/pybind/pybind11) (Wenzel Jakob) | headers compiled into the Python module | BSD-3-Clause | its notice must ship with the wheels (`LICENSES/BSD-3-Clause-pybind11.txt`; `setup.py` bundles `LICENSES/`) |
| OpenCL headers and ICD loader (Khronos) | headers at build time; the loader `OpenCL.dll` bundled in the Windows wheels | Apache-2.0 | elsewhere the ICD loader (`libOpenCL`, ocl-icd / Khronos) is linked dynamically and not bundled |
| GCC runtime: libstdc++, libgcc, libgomp | statically linked (`TN_STATIC_LINK`, MinGW), or bundled DLLs in Windows wheels | GPL-3.0 with the GCC Runtime Library Exception | the exception permits this for any license |
| mingw-w64 winpthreads (`libwinpthread-1.dll`) | bundled in Windows wheels / static in the Windows binary | MIT / BSD-style | notice to ship with Windows binaries |
| LLVM OpenMP (`libomp`) | macOS binary (static) / wheels (bundled by delocate) | Apache-2.0 WITH LLVM-exception | |
| MATLAB MEX API (`mex.h`, libmx / libmex) | the MATLAB MEX | proprietary (MathWorks) | GPL MEX files linking MATLAB's libraries are common practice (MATLAB as the system library); worth keeping in mind for redistribution |
| GNU Octave (liboctave) | the Octave MEX | GPL-3.0-or-later | compatible |
| NumPy | a runtime dependency of the Python package (not bundled) | BSD-3-Clause | |
| [jdata](https://github.com/NeuroJSON/pyjdata) (>= 0.9.5) | the Python tests only | Apache-2.0 | |

## Algorithms and methods to cite

- DistMesh: P.-O. Persson, G. Strang, "A simple mesh generator in MATLAB",
  *SIAM Review* 46(2), 329-345 (2004).
- The moving-particle (truss) mesher this re-develops: Q. Fang, SPIE 2006; R.
  Walton, MS thesis (2026).
- Constrained Delaunay tetrahedrization: L. Diazzi, D. Panozzo, A. Vaxman, M.
  Attene, "Constrained Delaunay Tetrahedrization: A Robust and Practical Approach",
  *ACM Trans. Graph.* 42(6) (SIGGRAPH Asia 2023).
- Indirect predicates: M. Attene, "Indirect Predicates for Geometric Constructions",
  *Computer-Aided Design* 126 (2020).
- Adaptive exact predicates: J. R. Shewchuk, *Discrete Comput. Geom.* 18, 305-363 (1997).
- GPU Delaunay refinement (the radius-edge test): gQM3d, Z. Chen, T.-S. Tan (NUS).
- brain2mesh / iso2mesh (tissue models, the TPM and gray-scale workflows):
  A. P. Tran, S. Yan, Q. Fang, *Neurophotonics* 7(1), 015008 (2020).
- Formats: JMesh / JNIfTI / JData (NeuroJSON, Q. Fang).

Test and benchmark data (Colin27, Digimouse, the ANTS atlas TPMs of the brain2mesh
sample, the siamize TPMs) are not part of this repository; their own terms apply
where they are used.
