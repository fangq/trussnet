// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_particles.h -- stages 3-9: graded hex seeding, the multi-level hash, the
// Verlet K-nearest truss, and the force / move / trap relaxation loop
// (src/opencl/v2m_seed_body.cl, v2m_particle_body.cl).

#ifndef V2MESH_PARTICLES_H
#define V2MESH_PARTICLES_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "v2m_grid.h"

namespace tn {

struct RelaxParams {
    int nseed = 8;          // coarse seeding levels (contiguous lattices)
    float t = 1.3f;         // truss reach: bar if |xi - xj| < t h_mid
    float skin = 0.3f;      // Verlet skin (fraction of h)
    float fscale = 1.2f;    // rest length l0 = fscale h_mid (DistMesh internal pressure)
    float fsurf = 1.0f;     // rest length / h of bars between two interface nodes
    float dt = 0.5f;        // Jacobi relaxation factor: step = dt * F / (active bars) (FIRE: its first step)
    float maxstep = 0.2f;   // step cap (fraction of h)
    float snap = 0.5f;      // interior nodes closer than snap*h to an interface join it
    int max_iters = 500;
    float dptol = 2e-3f;    // stop when the 99th percentile of |dp|/h < dptol
    float jseed = 0.8f;     // junction-line seeds: one per cell of jseed * level spacing (0 = off)
    bool corners = true;    // fixed CORNER nodes where >= 4 labels meet
    bool voxel_trap = false; // trap on voxel faces instead of the smooth interface
    float thin = 0.0f;      // seed thinning: drop seeds closer than thin*h to a kept one (0 = off)
    // FIRE integrator (inertial, adaptive step; default) or the Jacobi step: FIRE
    // reaches the quality Jacobi has after ~1000 iterations in ~300 (Colin27)
    bool fire = true;
    float fire_dtmax = 2.0f; // FIRE: largest time step, x the first (sqrt(dt): the Jacobi step)
    bool verbose = false;
};

// FIRE's global part (Bitzek et al. 2006): from the power P = sum_i a_i . v_i of
// the last step, grow the time step and relax the steering while the system goes
// downhill; halve it, reset the steering and stop every node when it goes uphill.
// The per-node part is in v2m_move (src/opencl/v2m_particle_body.cl).
struct FireCtl {
    float dt0, dt, dtmax, alpha;
    int npos = 0, resets = 0;
    explicit FireCtl(const RelaxParams& p)
        : dt0(std::sqrt(p.dt)), dt(dt0), dtmax(p.fire_dtmax * dt0), alpha(kAlpha0) {}
    // small moves are convergence only at a time step that has not collapsed
    // (uphill resets halve it; tiny steps would pass for settled nodes)
    bool may_stop() const {
        return dt >= 0.25f * dt0;
    }
    // true: zero every velocity
    bool update(double P) {
        if (P >= 0.0) {
            if (++npos > kNmin) {
                dt = std::min(dt * kInc, dtmax);
                alpha *= kFa;
            }

            return false;
        }

        npos = 0;
        dt = std::max(dt * kDec, 0.05f * dt0);
        alpha = kAlpha0;
        ++resets;
        return true;
    }
    static constexpr int kNmin = 5;
    static constexpr float kInc = 1.1f, kDec = 0.5f, kAlpha0 = 0.1f, kFa = 0.99f;
};

struct Nodes {
    std::vector<float> P;       // 3 per node, grid mm
    std::vector<uint16_t> lab;  // own label
    std::vector<uint8_t> typ;   // V2M_INTERIOR / INTERFACE / JUNCTION / CORNER
    std::vector<uint16_t> part; // 2 per node: partner labels
    std::vector<uint16_t> part3; // 1 per node: the 4th label of a CORNER node (host only)
    size_t size() const {
        return lab.size();
    }
};

struct RelaxStats {
    int iters = 0, rebuilds = 0;
    float last_move = 0, last_p99 = 0;
    int fire_resets = 0;   // FIRE: uphill steps (every velocity zeroed)
    float fire_dt = 0;     // FIRE: the time step at the end
    size_t n_interior = 0, n_interface = 0, n_junction = 0, n_corner = 0;
    double ms_seed = 0, ms_hash = 0, ms_force = 0, ms_move = 0;
};

void seed_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd);
void relax_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st);
// Remove crowded nodes (see v2m_particles.cpp): returns how many were removed.
size_t thin_nodes(const Grid& g, const RelaxParams& prm, Nodes& nd);
// Shape input: empty the diametral ball of each pair of neighbouring pinned
// nodes on a feature curve (g.feat), so the Delaunay keeps the pair joined
// (see v2m_particles.cpp): returns how many nodes were removed.
size_t protect_features(const Grid& g, Nodes& nd);

}  // namespace tn

#endif  // V2MESH_PARTICLES_H
