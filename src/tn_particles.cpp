// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_particles.cpp -- see tn_particles.h. Host (OpenMP) driver of the seeding and
// particle bodies; the round structure matches the device path.

#include "tn_particles.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#include "tn_log.h"
#include "tn_omp.h"

namespace tn {

namespace particle_host {

using std::ceil;
using std::exp;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::log;
using std::pow;
using std::sqrt;
typedef uint8_t uchar;
typedef uint16_t ushort;
#define TN_G
#include "opencl/tn_grid_body.cl"
#include "opencl/tn_seed_body.cl"
#include "opencl/tn_particle_body.cl"
#undef TN_G

}  // namespace particle_host

using namespace particle_host;

namespace {

TnDims dims_of(const Grid& g) {
    TnDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
    return d;
}

typedef std::chrono::steady_clock clk;
double since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

}  // namespace

#define GRID_FIELD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data()

void seed_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd) {
    OmpThreadCap cap;
    const TnDims d = dims_of(g);
    const int64_t nv = static_cast<int64_t>(g.nx) * g.ny * g.nz;
    std::vector<int> cnt(static_cast<size_t>(nv) + 1, 0);
    #pragma omp parallel for schedule(dynamic, 1024)

    for (int64_t v = 0; v < nv; ++v) {
        const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
        cnt[v + 1] = tn_seed_voxel(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.voxel_trap ? 1 : 0, i,
                                   j, k, 0, nullptr, nullptr, 0);
    }

    for (int64_t v = 0; v < nv; ++v) {
        cnt[v + 1] += cnt[v];
    }

    const int n = cnt[nv];
    nd.P.assign(static_cast<size_t>(n) * 3, 0.0f);
    nd.lab.assign(n, 0);
    nd.typ.assign(n, TN_INTERIOR);
    nd.part.assign(static_cast<size_t>(n) * 2, TN_NOLAB);
    #pragma omp parallel for schedule(dynamic, 1024)

    for (int64_t v = 0; v < nv; ++v) {
        if (cnt[v + 1] == cnt[v]) {
            continue;
        }

        const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
        tn_seed_voxel(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.voxel_trap ? 1 : 0, i, j, k, 1,
                      nd.P.data(), nd.lab.data(), cnt[v]);
    }

    #pragma omp parallel for schedule(dynamic, 1024)

    for (int i = 0; i < n; ++i) {
        tn_seed_classify(GRID_FIELD, g.h.data(), i, nd.P.data(), nd.lab.data(), nd.typ.data(), nd.part.data());
    }
}

void relax_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st) {
    OmpThreadCap cap;
    const TnDims d = dims_of(g);
    const int n = static_cast<int>(nd.size());

    // hash geometry: level-0 bin = the finest search radius
    TnHash H;
    H.b0 = (prm.t + prm.skin) * g.hmin;
    H.ox = -0.5f * g.vs[0];
    H.oy = -0.5f * g.vs[1];
    H.oz = -0.5f * g.vs[2];
    const float ext[3] = { g.nx * g.vs[0], g.ny * g.vs[1], g.nz * g.vs[2] };
    H.nlev = 1;

    while (H.nlev < TN_MAXLEV && H.b0 * static_cast<float>(1 << (H.nlev - 1)) < (prm.t + prm.skin) * g.hmax) {
        ++H.nlev;
    }

    int nkeys = 0;

    for (int L = 0; L < H.nlev; ++L) {
        H.off[L] = nkeys;
        const float bL = H.b0 * static_cast<float>(1 << L);

        for (int a = 0; a < 3; ++a) {
            H.dim[L][a] = std::max(1, static_cast<int>(std::ceil(ext[a] / bL)));
        }

        nkeys += H.dim[L][0] * H.dim[L][1] * H.dim[L][2];
    }

    std::vector<int> key(n), cstart(static_cast<size_t>(nkeys) + 1), sorted(n), nbr(static_cast<size_t>(n) * TN_K),
        nnb(n);
    std::vector<float> F(static_cast<size_t>(n) * 3), P0(nd.P), mv(n);

    auto rebuild = [&]() {
        clk::time_point t0 = clk::now();
        #pragma omp parallel for

        for (int i = 0; i < n; ++i) {
            const float h = tn_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
            key[i] = tn_bin_key(&H, tn_level_of(&H, (prm.t + prm.skin) * h), nd.P[3 * i], nd.P[3 * i + 1],
                                nd.P[3 * i + 2]);
        }

        std::fill(cstart.begin(), cstart.end(), 0);

        for (int i = 0; i < n; ++i) {
            ++cstart[key[i] + 1];
        }

        for (int k = 0; k < nkeys; ++k) {
            cstart[k + 1] += cstart[k];
        }

        std::vector<int> cur(cstart.begin(), cstart.end() - 1);

        for (int i = 0; i < n; ++i) {
            sorted[cur[key[i]]++] = i;    // ascending i within a bin: deterministic
        }

        #pragma omp parallel for schedule(dynamic, 256)

        for (int i = 0; i < n; ++i) {
            tn_neighbors(&H, d, g.h.data(), nd.P.data(), nd.lab.data(), nd.typ.data(), cstart.data(), sorted.data(),
                         prm.t, prm.skin, i, nbr.data(), nnb.data());
        }

        P0 = nd.P;
        ++st.rebuilds;
        st.ms_hash += since(t0);
    };

    rebuild();

    for (int it = 0; it < prm.max_iters; ++it) {
        clk::time_point t0 = clk::now();
        #pragma omp parallel for schedule(dynamic, 1024)

        for (int i = 0; i < n; ++i) {
            tn_force(d, g.h.data(), nd.P.data(), nd.typ.data(), nbr.data(), nnb.data(), prm.fscale, prm.fsurf, i,
                     &F[3 * i]);
        }

        st.ms_force += since(t0);
        clk::time_point t1 = clk::now();
        float mmax = 0.0f;
        #pragma omp parallel for schedule(dynamic, 1024) reduction(max : mmax)

        for (int i = 0; i < n; ++i) {
            mv[i] = tn_move(GRID_FIELD, g.h.data(), F.data(), prm.dt, prm.maxstep, prm.snap, i, nd.P.data(), nd.lab.data(),
                            nd.typ.data(), nd.part.data());
            mmax = std::max(mmax, mv[i]);
        }

        st.ms_move += since(t1);
        st.iters = it + 1;
        st.last_move = mmax;

        if (prm.verbose && (it % 50 == 0)) {
            TN_FPRINTF(stderr, "[relax] iter %d: max move %.4g h\n", it, mmax);
        }

        if (mmax < prm.dptol) {
            break;
        }

        // Verlet criterion: rebuild once some node has moved skin/2 since the build
        bool need = false;
        #pragma omp parallel for reduction(|| : need)

        for (int i = 0; i < n; ++i) {
            const float ex = nd.P[3 * i] - P0[3 * i], ey = nd.P[3 * i + 1] - P0[3 * i + 1],
                        ez = nd.P[3 * i + 2] - P0[3 * i + 2];
            const float h = tn_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
            need = need || (ex * ex + ey * ey + ez * ez > 0.25f * prm.skin * prm.skin * h * h);
        }

        if (need) {
            rebuild();
        }
    }

    st.n_interior = st.n_interface = st.n_junction = st.n_corner = 0;

    for (uint8_t t : nd.typ) {
        st.n_interior += t == TN_INTERIOR;
        st.n_interface += t == TN_INTERFACE;
        st.n_junction += t == TN_JUNCTION;
        st.n_corner += t == TN_CORNER;
    }
}

}  // namespace tn
