// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_gpu.h -- OpenCL device path: the relaxation loop (hash, truss, force, move,
// trap, reductions) on the GPU, over the same bodies as the OpenMP reference.

#ifndef V2MESH_GPU_H
#define V2MESH_GPU_H

#include "v2m_grid.h"
#include "v2m_particles.h"

namespace tn {

// Relax on OpenCL device `device` (flat index, -1 = first GPU). Same contract as
// relax_cpu. Throws std::runtime_error if OpenCL is unavailable or fails.
void relax_cl(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st, int device);

}  // namespace tn

#endif  // V2MESH_GPU_H
