// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_grid.cpp -- see tn_grid.h. Host (OpenMP) driver of tn_grid_body.cl.

#include "tn_grid.h"

#include <chrono>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cstdio>
#include <cstdlib>

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

    const bool gprof = std::getenv("TN_GRID_PROFILE") != nullptr;
    auto gt0 = std::chrono::steady_clock::now();
    auto glap = [&](const char* what) {
        if (gprof) {
            const auto t = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[gridprof] %-12s %7.0f ms\n", what, std::chrono::duration<double, std::milli>(t - gt0).count());
            gt0 = t;
        }
    };

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

    glap("bricks+slots");
    // 3. smoothed indicators per slot: first with the wide curvature sigma (for
    // the sizing below), then overwritten with the interface sigma
    const size_t ns = g.slot_brick.size();
    g.phi.assign(ns * TN_SLOT, 0.0f);
    // separable Gaussian on the (8+2R)^3 tile of each slot's brick: the same
    // algorithm as g_smooth (the direct 13^3 sum took ~25 s on Colin27)
    auto smooth_all = [&](float sigma, int R) {
        float w[2 * 6 + 1], ws = 0.0f;

        for (int dd = -R; dd <= R; ++dd) {
            w[dd + R] = tn_gauss(dd, sigma);
            ws += w[dd + R];
        }

        for (int dd = 0; dd <= 2 * R; ++dd) {
            w[dd] /= ws;
        }

        const int T = TN_BS + 2 * R;
        #pragma omp parallel
        {
            std::vector<float> A(static_cast<size_t>(T) * T * T), B(A.size());
            #pragma omp for schedule(dynamic, 4)

            for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
                const int b = g.slot_brick[s];
                const int l = g.slot_label[s];
                const int x0 = (b % g.nbx) * TN_BS - R, y0 = ((b / g.nbx) % g.nby) * TN_BS - R,
                          z0 = (b / (g.nbx * g.nby)) * TN_BS - R;

                for (int z = 0; z < T; ++z)
                    for (int y = 0; y < T; ++y)
                        for (int x = 0; x < T; ++x) {
                            A[x + T * (y + T * z)] = tn_label_at(L, g.nx, g.ny, g.nz, x0 + x, y0 + y, z0 + z) == l ? 1.0f : 0.0f;
                        }

                for (int z = 0; z < T; ++z)   // x, on the centre columns only
                    for (int y = 0; y < T; ++y)
                        for (int x = R; x < R + TN_BS; ++x) {
                            float v = 0.0f;

                            for (int dd = -R; dd <= R; ++dd) {
                                v += w[dd + R] * A[x + dd + T * (y + T * z)];
                            }

                            B[x + T * (y + T * z)] = v;
                        }

                for (int z = 0; z < T; ++z)   // y
                    for (int y = R; y < R + TN_BS; ++y)
                        for (int x = R; x < R + TN_BS; ++x) {
                            float v = 0.0f;

                            for (int dd = -R; dd <= R; ++dd) {
                                v += w[dd + R] * B[x + T * (y + dd + T * z)];
                            }

                            A[x + T * (y + T * z)] = v;
                        }

                for (int t = 0; t < TN_SLOT; ++t) {   // z, centre
                    const int x = t % TN_BS + R, y = (t / TN_BS) % TN_BS + R, z = t / (TN_BS * TN_BS) + R;
                    float v = 0.0f;

                    for (int dd = -R; dd <= R; ++dd) {
                        v += w[dd + R] * A[x + T * (y + T * (z + dd))];
                    }

                    g.phi[s * TN_SLOT + t] = v;
                }
            }
        }
    };
    smooth_all(prm.sigma_curv, Rc);
    glap("smooth-curv");

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

    // local layer thickness (voxels) from the sigma_curv fields: for the thickness
    // sizing and the adaptive interface field
    std::vector<float> tvox;

    glap("sizing");

    if (prm.thick > 0.0f || prm.sigma_thin > 0.0f) {
        tvox.assign(nv, 1e30f);
        #pragma omp parallel for schedule(dynamic, 4096)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
            tvox[v] = tn_thick_voxel(d, L, g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(),
                                     prm.sigma_curv, 2, i, j, k);
        }
    }

    // thin layers: h <= t / thick, t from the (still sigma_curv) smoothed fields,
    // down to a floor of thin_floor voxels -- below the curvature hmin, which a
    // flat 1-2 voxel sheet never triggers
    if (prm.thick > 0.0f) {
        const float floor_mm = prm.thin_floor * vmin;
        #pragma omp parallel for schedule(dynamic, 4096)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
            const float t = tvox[v];
            (void)i;
            (void)j;
            (void)k;

            if (t < 1e29f) {
                g.h[v] = std::min(g.h[v], std::max(floor_mm, t * vmin / prm.thick));
            }
        }

        g.hmin = std::min(g.hmin, floor_mm);
    }

    glap("thickness");
    const bool gray = !lv.gray.empty() && !lv.thresholds.empty();
    g.gI = nullptr;
    g.gm = 0;
    g.gTW.assign(1, 0.0f);

    if (gray) {
        // gray-scale input: membership fields of the intensity instead of smoothed
        // indicators. s_k = (I - t_k) / W_k, phi_l = clamp(0.5 + min(s_{l-1}, -s_l));
        // near t_k, phi_{k-1} - phi_k = -2 s_k: psi is linear in the (trilinear)
        // intensity and vanishes exactly on the iso-surface I = t_k. W_k = 4 x the
        // median intensity step across the iso-surface, so the linear band covers
        // +-2 voxels around it (a trilinear cell next to it stays inside).
        const std::vector<float>& I = lv.gray;
        const std::vector<float>& T = lv.thresholds;
        const int m = static_cast<int>(T.size());
        std::vector<float> W(m, 1.0f);

        for (int k = 0; k < m; ++k) {
            std::vector<float> steps;
            const int64_t st[3] = { 1, g.nx, static_cast<int64_t>(g.nx) * g.ny };
            const int len[3] = { g.nx, g.ny, g.nz };

            for (int a = 0; a < 3; ++a)
                for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
                    if ((v / st[a]) % len[a] + 1 >= len[a]) {
                        continue;
                    }

                    const float x = I[v] - T[k], y = I[v + st[a]] - T[k];

                    if ((x < 0.0f) != (y < 0.0f)) {
                        steps.push_back(std::fabs(y - x));
                    }
                }

            if (!steps.empty()) {
                std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
                W[k] = std::max(1e-12f, 4.0f * steps[steps.size() / 2]);
            }
        }

        g.gray_w = W;
        g.gI = I.data();
        g.gm = m;
        g.gTW.assign(T.begin(), T.end());
        g.gTW.insert(g.gTW.end(), W.begin(), W.end());
        #pragma omp parallel for schedule(dynamic, 4)

        for (int64_t s2 = 0; s2 < static_cast<int64_t>(ns); ++s2) {
            const int b = g.slot_brick[s2];
            const int l = g.slot_label[s2];
            const int bx = b % g.nbx, by = (b / g.nbx) % g.nby, bz = b / (g.nbx * g.nby);

            for (int t = 0; t < TN_SLOT; ++t) {
                const int i = bx * TN_BS + t % TN_BS, j = by * TN_BS + (t / TN_BS) % TN_BS,
                          k = bz * TN_BS + t / (TN_BS * TN_BS);
                float f;

                if (i >= g.nx || j >= g.ny || k >= g.nz) {
                    f = l == 0 ? 1.0f : 0.0f;
                } else {
                    const float x = I[i + static_cast<size_t>(g.nx) * (j + static_cast<size_t>(g.ny) * k)];
                    float mm = 1e30f;

                    if (l >= 1 && l - 1 < m) {
                        mm = std::min(mm, (x - T[l - 1]) / W[l - 1]);
                    }

                    if (l < m) {
                        mm = std::min(mm, (T[l] - x) / W[l]);
                    }

                    f = std::min(1.0f, std::max(0.0f, 0.5f + mm));
                }

                g.phi[static_cast<size_t>(s2) * TN_SLOT + t] = f;
            }
        }
    } else {
        smooth_all(prm.sigma, Ri);   // the interface fields kept for trapping
    }

    if (!gray && prm.sigma_thin > 0.0f) {
        // blend weight per voxel: min thickness over a 5^3 neighbourhood (every label
        // at a point must see the same w), smoothstep, then a sigma = 1 blur so the
        // blended field has no kinks; separable passes on the dense grid
        std::vector<float> w(tvox), tmp(nv);
        auto pass = [&](std::vector<float>& src, std::vector<float>& dst, int axis, int r, bool mn, const float* ker) {
            const int64_t st = axis == 0 ? 1 : (axis == 1 ? g.nx : static_cast<int64_t>(g.nx) * g.ny);
            const int nlen = axis == 0 ? g.nx : (axis == 1 ? g.ny : g.nz);
            #pragma omp parallel for schedule(static)

            for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
                const int c = static_cast<int>((v / st) % nlen);
                float acc = mn ? 1e30f : 0.0f;

                for (int dd = -r; dd <= r; ++dd) {
                    const int cc = std::min(nlen - 1, std::max(0, c + dd));
                    const float x = src[v + (cc - c) * st];
                    acc = mn ? std::min(acc, x) : acc + ker[dd + r] * x;
                }

                dst[v] = acc;
            }
        };

        for (int a = 0; a < 3; ++a) {   // min filter, radius 2
            pass(w, tmp, a, 2, true, nullptr);
            w.swap(tmp);
        }

        for (size_t v = 0; v < nv; ++v) {
            float x = (w[v] - prm.thin_lo) / std::max(1e-3f, prm.thin_hi - prm.thin_lo);
            x = std::min(1.0f, std::max(0.0f, x));
            w[v] = x * x * (3.0f - 2.0f * x);
        }

        float ker[7], ks = 0.0f;

        for (int dd = -3; dd <= 3; ++dd) {
            ker[dd + 3] = tn_gauss(dd, 1.0f);
            ks += ker[dd + 3];
        }

        for (float& x : ker) {
            x /= ks;
        }

        for (int a = 0; a < 3; ++a) {
            pass(w, tmp, a, 3, false, ker);
            w.swap(tmp);
        }

        std::vector<float> phi_s(g.phi);
        const int Rt = std::min(6, std::max(1, static_cast<int>(std::ceil(3.0f * prm.sigma_thin))));
        smooth_all(prm.sigma_thin, Rt);   // g.phi <- the sharp field
        #pragma omp parallel for schedule(dynamic, 4)

        for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
            const int b = g.slot_brick[s];
            const int bx = b % g.nbx, by = (b / g.nbx) % g.nby, bz = b / (g.nbx * g.nby);

            for (int t = 0; t < TN_SLOT; ++t) {
                const int i = bx * TN_BS + t % TN_BS, j = by * TN_BS + (t / TN_BS) % TN_BS,
                          k = bz * TN_BS + t / (TN_BS * TN_BS);

                if (i >= g.nx || j >= g.ny || k >= g.nz) {
                    continue;
                }

                const float wv = w[i + static_cast<size_t>(g.nx) * (j + static_cast<size_t>(g.ny) * k)];
                const size_t o = static_cast<size_t>(s) * TN_SLOT + t;
                g.phi[o] = wv * phi_s[o] + (1.0f - wv) * g.phi[o];
            }
        }
    }

    glap("interface");

    if (!gray && prm.preserve > 0.0f) {   // keep every voxel centre's own label on top
        #pragma omp parallel for schedule(dynamic, 4096)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
            tn_preserve_voxel(d, L, g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), prm.preserve, i,
                              j, k);
        }
    }

    // 5. gradient limiting (Jacobi sweeps until nothing changes)
    // Frontier Jacobi: the first sweep covers every voxel; afterwards only the 26
    // neighbours of voxels that changed are evaluated (any other voxel reads only
    // unchanged values and cannot change), still reading the previous sweep's
    // values -- the same result as full sweeps. The frontier is a thin moving band,
    // so the ~20 sweeps no longer each cost a pass over the whole volume.
    std::vector<float> hn(g.h);
    std::vector<char> chg(nv, 0), mark(nv, 0);
    std::vector<int64_t> active;
    g.limit_sweeps = 0;

    for (;;) {
        const bool full = g.limit_sweeps == 0;
        const int64_t na = full ? static_cast<int64_t>(nv) : static_cast<int64_t>(active.size());
        int changed = 0;
        #pragma omp parallel for schedule(dynamic, 4096) reduction(| : changed)

        for (int64_t a = 0; a < na; ++a) {
            const int64_t v = full ? a : active[a];
            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
            chg[v] = static_cast<char>(tn_limit_voxel(d, g.h.data(), hn.data(), prm.g, i, j, k));
            changed |= chg[v];
        }

        ++g.limit_sweeps;
        #pragma omp parallel for schedule(static)

        for (int64_t a = 0; a < na; ++a) {   // commit the sweep (Jacobi) and mark the frontier
            const int64_t v = full ? a : active[a];
            g.h[v] = hn[v];

            if (!chg[v]) {
                continue;
            }

            const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                      k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));

            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = i + dx, y = j + dy, z = k + dz;

                        if (x >= 0 && y >= 0 && z >= 0 && x < g.nx && y < g.ny && z < g.nz) {
                            mark[x + static_cast<size_t>(g.nx) * (y + static_cast<size_t>(g.ny) * z)] = 1;   // benign
                        }
                    }
        }

        if (!changed) {
            break;
        }

        #pragma omp parallel for schedule(static)

        for (int64_t a = 0; a < na; ++a) {
            chg[full ? a : active[a]] = 0;
        }

        active.clear();   // gather the marked voxels (ascending), clearing the marks
        std::vector<std::vector<int64_t>> part;
#ifdef _OPENMP
        part.resize(static_cast<size_t>(omp_get_max_threads()));
#else
        part.resize(1);
#endif
        #pragma omp parallel
        {
#ifdef _OPENMP
            std::vector<int64_t>& mine = part[static_cast<size_t>(omp_get_thread_num())];
#else
            std::vector<int64_t>& mine = part[0];
#endif
            #pragma omp for schedule(static)

            for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v)
                if (mark[v]) {
                    mine.push_back(v);
                    mark[v] = 0;
                }
        }

        for (auto& q : part) {
            active.insert(active.end(), q.begin(), q.end());
        }
    }

    glap("limiting");
    // 6. grades
    g.grade.resize(nv);
    #pragma omp parallel for

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        g.grade[v] = tn_grade(g.h[v], g.hmin, g.hmax);
    }
}

}  // namespace tn
