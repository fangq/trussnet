// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_particles.h -- stages 3-9: graded hex seeding, the multi-level hash, the
// Verlet K-nearest truss, and the force / move / trap relaxation loop
// (src/opencl/tn_seed_body.cl, tn_particle_body.cl).

#ifndef TRUSSNET_PARTICLES_H
#define TRUSSNET_PARTICLES_H

#include <cstdint>
#include <vector>

#include "tn_grid.h"

namespace tn {

struct RelaxParams {
    int nseed = 8;          // coarse seeding levels (contiguous lattices)
    float t = 1.3f;         // truss reach: bar if |xi - xj| < t h_mid
    float skin = 0.3f;      // Verlet skin (fraction of h)
    float fscale = 1.2f;    // rest length l0 = fscale h_mid (DistMesh internal pressure)
    float fsurf = 1.0f;     // rest length / h of bars between two interface nodes
    float dt = 0.5f;        // Jacobi relaxation factor: step = dt * F / (active bars)
    float maxstep = 0.2f;   // step cap (fraction of h)
    float snap = 0.5f;      // interior nodes closer than snap*h to an interface join it
    int max_iters = 500;
    float dptol = 2e-3f;    // stop when the 99th percentile of |dp|/h < dptol
    float jseed = 0.8f;     // junction-line seeds: one per cell of jseed * level spacing (0 = off)
    bool corners = true;    // fixed CORNER nodes where >= 4 labels meet
    bool voxel_trap = false; // trap on voxel faces instead of the smooth interface
    float thin = 0.0f;      // seed thinning: drop seeds closer than thin*h to a kept one (0 = off)
    bool verbose = false;
};

struct Nodes {
    std::vector<float> P;       // 3 per node, grid mm
    std::vector<uint16_t> lab;  // own label
    std::vector<uint8_t> typ;   // TN_INTERIOR / INTERFACE / JUNCTION / CORNER
    std::vector<uint16_t> part; // 2 per node: partner labels
    std::vector<uint16_t> part3; // 1 per node: the 4th label of a CORNER node (host only)
    size_t size() const {
        return lab.size();
    }
};

struct RelaxStats {
    int iters = 0, rebuilds = 0;
    float last_move = 0, last_p99 = 0;
    size_t n_interior = 0, n_interface = 0, n_junction = 0, n_corner = 0;
    double ms_seed = 0, ms_hash = 0, ms_force = 0, ms_move = 0;
};

void seed_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd);
void relax_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st);
// Remove crowded nodes (see tn_particles.cpp): returns how many were removed.
size_t thin_nodes(const Grid& g, const RelaxParams& prm, Nodes& nd);

}  // namespace tn

#endif  // TRUSSNET_PARTICLES_H
