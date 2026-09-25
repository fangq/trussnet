// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_gdel.h -- the Delaunay tetrahedrization on the OpenCL device (rounds of
// parallel Bowyer-Watson insertion, opencl/tn_del_kernels.cl), loaded into the
// vendored Diazzi TetMesh so the incremental repairs keep working on it. Points
// whose predicates the device filters cannot certify are inserted afterwards by
// the exact CPU code: the result is exactly TetMesh::tetrahedrize()'s.

#ifndef TRUSSNET_GDEL_H
#define TRUSSNET_GDEL_H

#include <cstddef>
#include <cstdint>

class TetMesh;

namespace tn {

struct GdelStats {
    int rounds = 0;
    size_t sampled = 0;        // points of the initial CPU sample
    size_t gpu_inserted = 0;   // points inserted on the device
    size_t deferred = 0;       // points left to the exact CPU insertion
    size_t dead = 0;           // surplus cavity slots compacted away
    double ms_build = 0, ms_gpu = 0, ms_load = 0, ms_cpu = 0;
};

// Fill the FRESH `tm` (no vertices yet) with the Delaunay tetrahedrization of the
// n points X (xyz interleaved). Returns false (tm untouched) if the device path is
// unavailable or fails; the caller then runs tm.tetrahedrize() itself.
bool gdel_tetrahedrize(const double* X, uint32_t n, ::TetMesh& tm, GdelStats& st, int device);

// Put a TetMesh (no deleted tets) in a canonical form that depends only on the
// triangulation: tets ordered by their sorted corners, the corners of each rotated
// by an even permutation (orientation kept) to start at the smallest, the infinite
// vertex staying last. The device build's slot order varies from run to run; after
// this the CPU and the device paths hand the same mesh to the repairs.
void canonicalize_tets(::TetMesh& tm);

}  // namespace tn

#endif  // TRUSSNET_GDEL_H
