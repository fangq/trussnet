// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_grid.cpp -- see tn_grid.h. Host (OpenMP) driver of tn_grid_body.cl.

#include "tn_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tn_omp.h"

namespace tn {

namespace grid_host {

using std::exp;
using std::fabs;
using std::floor;
using std::fmin;
using std::log;
using std::pow;
using std::sqrt;
typedef uint8_t uchar;
typedef uint16_t ushort;
#define TN_G
#include "opencl/tn_grid_body.cl"
#undef TN_G

}  // namespace grid_host

using namespace grid_host;

void build_grid_cpu(const LabelVolume& lv, const GridParams& prm, Grid& g) {
    OmpThreadCap cap;
    g.nx = lv.nx;
    g.ny = lv.ny;
    g.nz = lv.nz;
    g.nbx = (g.nx + TN_BS - 1) / TN_BS;
    g.nby = (g.ny + TN_BS - 1) / TN_BS;
    g.nbz = (g.nz + TN_BS - 1) / TN_BS;

    for (int k = 0; k < 3; ++k) {
        g.vs[k] = static_cast<float>(lv.voxelsize[k]);
    }

    g.sigma = prm.sigma;
    const int Ri = std::min(6, std::max(1, static_cast<int>(std::ceil(3.0f * prm.sigma))));
    const int Rc = std::min(6, std::max(1, static_cast<int>(std::ceil(3.0f * prm.sigma_curv))));
    g.R = std::max(Ri, Rc);
    g.L = &lv.data;
    g.nlab = lv.maxlabel + 1;
    const float vmin = std::min(g.vs[0], std::min(g.vs[1], g.vs[2]));
    g.hbase = prm.hbase > 0 ? prm.hbase : 3.0f * vmin;
    g.hmin = prm.hmin > 0 ? prm.hmin : g.hbase / 3.0f;
    g.hmax = prm.hmax > 0 ? prm.hmax : g.hbase;

    for (float v : prm.hlab)
        if (v > 0) {
            g.hmax = std::max(g.hmax, v);
            g.hmin = std::min(g.hmin, v);
        }

    const int nb = g.nbx * g.nby * g.nbz;
    const uint16_t* L = lv.data.data();

    // 1. labels near each brick
    g.bl_cnt.assign(nb, 0);
    g.bl_lab.assign(static_cast<size_t>(nb) * TN_BL, 0xFFFF);
    #pragma omp parallel for schedule(dynamic, 64)

    for (int b = 0; b < nb; ++b) {
        ushort lab[TN_BL];
        const int n = tn_brick_labels(L, g.nx, g.ny, g.nz, b % g.nbx, (b / g.nbx) % g.nby, b / (g.nbx * g.nby), g.R,
                                      lab);
        g.bl_cnt[b] = n;

        for (int i = 0; i < TN_BL && i < n; ++i) {
            g.bl_lab[static_cast<size_t>(b) * TN_BL + i] = lab[i];
        }
    }

    // 2. slots for mixed bricks (scan)
    g.bl_slot.assign(nb, -1);
    g.slot_brick.clear();
    g.slot_label.clear();
    g.overflow_bricks = 0;

    for (int b = 0; b < nb; ++b) {
        const int n = std::min(g.bl_cnt[b], TN_BL);
        g.overflow_bricks += g.bl_cnt[b] > TN_BL;

        if (n <= 1) {
            continue;
        }

        g.bl_slot[b] = static_cast<int>(g.slot_brick.size());

        for (int i = 0; i < n; ++i) {
            g.slot_brick.push_back(b);
            g.slot_label.push_back(g.bl_lab[static_cast<size_t>(b) * TN_BL + i]);
        }
    }

    // 3. smoothed indicators per slot: first with the wide curvature sigma (for
    // the sizing below), then overwritten with the interface sigma
    const size_t ns = g.slot_brick.size();
    g.phi.assign(ns * TN_SLOT, 0.0f);
    auto smooth_all = [&](float sigma, int R) {
        #pragma omp parallel for schedule(dynamic, 4)

        for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
            const int b = g.slot_brick[s];
            const int l = g.slot_label[s];
            const int bx = b % g.nbx, by = (b / g.nbx) % g.nby, bz = b / (g.nbx * g.nby);

            for (int t = 0; t < TN_SLOT; ++t) {
                const int i = bx * TN_BS + t % TN_BS, j = by * TN_BS + (t / TN_BS) % TN_BS,
                          k = bz * TN_BS + t / (TN_BS * TN_BS);
                g.phi[s * TN_SLOT + t] = tn_smooth_voxel(L, g.nx, g.ny, g.nz, i, j, k, l, sigma, R);
            }
        }
    };
    smooth_all(prm.sigma_curv, Rc);

    // 4. sizing per voxel
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
    const size_t nv = static_cast<size_t>(g.nx) * g.ny * g.nz;
    std::vector<float> hlab(std::max(1, g.nlab), 0.0f);

    for (size_t l = 0; l < prm.hlab.size() && l < hlab.size(); ++l) {
        hlab[l] = prm.hlab[l];
    }

    g.h.assign(nv, g.hmax);
    #pragma omp parallel for schedule(dynamic, 4096)

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
        g.h[v] = tn_size_voxel(d, L, g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), hlab.data(),
                               static_cast<int>(hlab.size()), g.hbase, g.hmin, g.hmax, prm.K, i, j, k);
    }

    smooth_all(prm.sigma, Ri);   // the interface fields kept for trapping

    // 5. gradient limiting (Jacobi sweeps until nothing changes)
    std::vector<float> hn(nv);
    g.limit_sweeps = 0;

    for (;;) {
        int changed = 0;
        #pragma omp parallel for schedule(dynamic, 4096) reduction(| : changed)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
            changed |= tn_limit_voxel(d, g.h.data(), hn.data(), prm.g, i, j, k);
        }

        g.h.swap(hn);
        ++g.limit_sweeps;

        if (!changed) {
            break;
        }
    }

    // 6. grades
    g.grade.resize(nv);
    #pragma omp parallel for

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        g.grade[v] = tn_grade(g.h[v], g.hmin, g.hmax);
    }
}

}  // namespace tn
