// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_gdel.cpp -- host driver of the device Delaunay (see tn_gdel.h and
// opencl/tn_del_kernels.cl): the Diazzi base tet on the host, rounds of
// locate / pick / cavity / check / commit / reset on the device with one small
// read-back per round, then the arrays are loaded into the TetMesh and the
// deferred points are inserted exactly on the CPU.
// TN_GDEL_CHECK=1 also runs TetMesh::tetrahedrize() and compares the tet sets.

#include "tn_gdel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "delaunay.h"
#include "tn_cl_host.h"

namespace tn {

namespace {

typedef std::chrono::steady_clock clk;
double since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

const uint32_t NONE = 0xFFFFFFFFu, DEAD = 0xFFFFFFFEu;
const uint32_t MAXC = 128, MAXB = 256;   // = tn_del_kernels.cl

std::string slurp(const std::string& path) {
    std::ifstream f(path);

    if (!f) {
        throw std::runtime_error("cannot read OpenCL source " + path);
    }

    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct Kern {
    cl_kernel k = nullptr;
    int na = 0;
    template <typename T>
    Kern& a(const T& v) {
        cl_check(clSetKernelArg(k, na++, sizeof(T), &v), "clSetKernelArg");
        return *this;
    }
    void run(cl_command_queue q, size_t n, size_t ls = 128) {
        const size_t gs = ((n + ls - 1) / ls) * ls;

        if (gs) {
            cl_check(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gs, &ls, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        }

        na = 0;
    }
};

struct Prog {
    cl_program prog = nullptr;
    Kern locate, pick, unpick, cavity, claim, check, fix, clear, commit, reset;
};

// built once per process on the shared context
Prog& program(ClCtx& ctx) {
    static Prog P;

    if (!P.prog) {
#ifdef TN_SRC_DIR
        const std::string dir = std::string(TN_SRC_DIR) + "/opencl/";
#else
        const std::string dir = "src/opencl/";
#endif
        const std::string src = slurp(dir + "tn_del_body.cl") + slurp(dir + "tn_del_kernels.cl");
        P.prog = ctx.build("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#pragma OPENCL FP_CONTRACT OFF\n" + src,
                           std::string("-cl-std=CL1.2") + (std::getenv("TN_GDEL_OPTS") ? std::string(" ") + std::getenv("TN_GDEL_OPTS") : ""));
        cl_int e;
        auto mk = [&](const char* name) {
            Kern k;
            k.k = clCreateKernel(P.prog, name, &e);
            cl_check(e, name);
            return k;
        };
        P.locate = mk("d_locate");
        P.pick = mk("d_pick");
        P.unpick = mk("d_unpick");
        P.cavity = mk("d_cavity");
        P.claim = mk("d_claim");
        P.check = mk("d_check");
        P.fix = mk("d_fix");
        P.clear = mk("d_clear");
        P.commit = mk("d_commit");
        P.reset = mk("d_reset");
    }

    return P;
}

void fill_u32(ClCtx& ctx, cl_mem b, uint32_t v, size_t off, size_t count) {
    if (count) {
        cl_check(clEnqueueFillBuffer(ctx.queue(), b, &v, 4, off * 4, count * 4, 0, nullptr, nullptr), "fill");
    }
}

// grow a u32 buffer (old contents kept, the tail set to `tail`)
cl_mem grow(ClCtx& ctx, cl_mem old, size_t oldn, size_t newn, uint32_t tail, bool fill_tail) {
    cl_mem b = ctx.alloc(newn * 4);
    cl_check(clEnqueueCopyBuffer(ctx.queue(), old, b, 0, 0, oldn * 4, 0, nullptr, nullptr), "copy");

    if (fill_tail) {
        fill_u32(ctx, b, tail, oldn, newn - oldn);
    }

    clReleaseMemObject(old);
    return b;
}

}  // namespace

void canonicalize_tets(::TetMesh& tm) {
    const int64_t nt = tm.numTets();
    const uint32_t nv = tm.numVertices();
    std::vector<uint8_t> pos(static_cast<size_t>(nt) * 4);   // pos[4t+i]: the new place of old corner i
    std::vector<std::array<uint32_t, 4>> key(nt);
    // 1. per tet: the even permutation and the sort key
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t) {
        const uint32_t* v = &tm.tet_node[4 * t];
        int o[4] = { 0, 1, 2, 3 };   // o[new position] = old corner

        if (v[3] == INFINITE_VERTEX) {   // cyclic on 0..2 (even), INF stays at 3
            int m = 0;

            for (int i = 1; i < 3; ++i)
                if (v[i] < v[m]) {
                    m = i;
                }

            o[0] = m;
            o[1] = (m + 1) % 3;
            o[2] = (m + 2) % 3;
        } else {
            int m = 0;

            for (int i = 1; i < 4; ++i)
                if (v[i] < v[m]) {
                    m = i;
                }

            if (m != 0) {   // double transposition (0 m)(the other two): even
                int r[2], k = 0;

                for (int i = 1; i < 4; ++i)
                    if (i != m) {
                        r[k++] = i;
                    }

                o[0] = m;
                o[m] = 0;
                o[r[0]] = r[1];
                o[r[1]] = r[0];
            }

            // then a cyclic rotation of positions 1..3 (even) to put the smallest next
            int m1 = 1;

            for (int i = 2; i < 4; ++i)
                if (v[o[i]] < v[o[m1]]) {
                    m1 = i;
                }

            const int a = o[1], b = o[2], c = o[3];

            if (m1 == 2) {
                o[1] = b;
                o[2] = c;
                o[3] = a;
            } else if (m1 == 3) {
                o[1] = c;
                o[2] = a;
                o[3] = b;
            }
        }

        for (int i = 0; i < 4; ++i) {
            pos[4 * t + o[i]] = static_cast<uint8_t>(i);
            key[t][i] = v[o[i]];
        }
    }

    // 2. order: counting sort by the first corner, then the (few) tets of a bucket
    // by the full key
    std::vector<int64_t> start(static_cast<size_t>(nv) + 2, 0);

    for (int64_t t = 0; t < nt; ++t) {
        ++start[key[t][0] + 1];
    }

    for (size_t i = 1; i < start.size(); ++i) {
        start[i] += start[i - 1];
    }

    std::vector<uint32_t> order(nt);
    {
        std::vector<int64_t> cur(start.begin(), start.end() - 1);

        for (int64_t t = 0; t < nt; ++t) {
            order[cur[key[t][0]]++] = static_cast<uint32_t>(t);
        }
    }
    #pragma omp parallel for schedule(dynamic, 1024)

    for (int64_t b = 0; b < static_cast<int64_t>(nv); ++b) {
        std::sort(order.begin() + start[b], order.begin() + start[b + 1],
                  [&](uint32_t x, uint32_t y) {
                      return key[x] < key[y];
                  });
    }

    std::vector<uint32_t> rank(nt);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < nt; ++i) {
        rank[order[i]] = static_cast<uint32_t>(i);
    }

    // 3. rewrite
    std::vector<uint32_t> node(static_cast<size_t>(nt) * 4);
    std::vector<uint64_t> neigh(static_cast<size_t>(nt) * 4);
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t) {
        const uint64_t r = rank[t];

        for (int i = 0; i < 4; ++i) {
            const uint64_t c = tm.tet_neigh[4 * t + i], u = c >> 2;
            node[4 * r + pos[4 * t + i]] = tm.tet_node[4 * t + i];
            neigh[4 * r + pos[4 * t + i]] = (static_cast<uint64_t>(rank[u]) << 2) | pos[c];
        }
    }

    tm.tet_node.swap(node);
    tm.tet_neigh.swap(neigh);
    std::fill(tm.mark_tetrahedra.begin(), tm.mark_tetrahedra.end(), 0);

    for (int64_t t = nt - 1; t >= 0; --t) {   // (the lowest real tet of each vertex)
        if (tm.tet_node[4 * t + 3] != INFINITE_VERTEX) {
            for (int i = 0; i < 4; ++i) {
                tm.inc_tet[tm.tet_node[4 * t + i]] = static_cast<uint64_t>(t);
            }
        }
    }
}

bool gdel_tetrahedrize(const double* X, uint32_t n, ::TetMesh& tm, GdelStats& st, int device) {
    if (n < 64) {
        return false;
    }

    const bool verbose = std::getenv("TN_GDEL_VERBOSE") != nullptr;
    clk::time_point t0 = clk::now();
    ClCtx* pctx;
    Prog* pp;

    try {
        pctx = &cl_shared_ctx(device);
        pp = &program(*pctx);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[gdel] device Delaunay unavailable (%s): CPU\n", e.what());
        return false;
    }

    ClCtx& ctx = *pctx;
    Prog& P = *pp;
    cl_command_queue q = ctx.queue();
    st.ms_build = since(t0);
    t0 = clk::now();

    tm.init_vertices(X, n);
    // the base tet of TetMesh::init(), on the ORIGINAL indices (no swap needed: the
    // symbolic perturbation only sees vertex ids)
    uint32_t i = 0, j = 1, k = 2, l = 3;
    int ori = 0;

    for (k = 2; ori == 0 && k < n - 1; k++)
        for (l = k + 1; ori == 0 && l < n; l++) {
            ori = tm.vOrient3D(i, j, k, l);
        }

    l--;
    k--;

    if (ori == 0) {
        return false;
    }

    if (ori < 0) {
        std::swap(i, j);
    }

    const uint32_t I = NONE;
    const uint32_t base_tet[20] = { l, k, j, i, l, j, k, I, l, k, i, I, l, i, j, I, k, j, i, I };
    const uint64_t base_neigh[20] = { 19, 15, 11, 7, 18, 10, 13, 3, 17, 14, 5, 2, 16, 6, 9, 1, 12, 8, 4, 0 };
    tm.resizeTets(5);
    std::memcpy(tm.tet_node.data(), base_tet, sizeof base_tet);
    std::memcpy(tm.tet_neigh.data(), base_neigh, sizeof base_neigh);
    tm.inc_tet[i] = tm.inc_tet[j] = tm.inc_tet[k] = tm.inc_tet[l] = 0;

    std::vector<uint8_t> state(n, 0);
    state[i] = state[j] = state[k] = state[l] = 1;
    // a strided sample first, exactly on the CPU: with only a few points in, every
    // pending point is outside the hull and all their cavities share the ghost tets,
    // so the device rounds would insert ~1 point each; the sample makes the hull
    // (and the cavities) local from the first device round on
    {
        static const uint32_t div = std::getenv("TN_GDEL_SAMPLE") ? std::atoi(std::getenv("TN_GDEL_SAMPLE")) : 256;
        static const uint32_t mmax = std::getenv("TN_GDEL_SAMPLE_MAX") ? std::atoi(std::getenv("TN_GDEL_SAMPLE_MAX")) : 8192;
        const uint32_t m = std::min<uint32_t>(n, std::max<uint32_t>(512, std::min<uint32_t>(n / div, mmax)));
        const uint32_t stride = std::max<uint32_t>(1, n / m);
        uint64_t ct = 0;

        for (uint32_t v = 0; v < n; v += stride) {
            if (!state[v]) {
                tm.insertExistingVertex(v, ct);
                state[v] = 1;
                ++st.sampled;
            }
        }

        tm.removeDelTets();
    }

    size_t cap = static_cast<size_t>(n) * 7 + 4096;
    const uint32_t maxcand = std::min<uint32_t>(n / 2 + 1024, 1u << 18);
    const uint32_t nt_init = tm.numTets();
    std::vector<uint32_t> neigh32(tm.tet_neigh.begin(), tm.tet_neigh.end());

    cl_mem dX = ctx.alloc(static_cast<size_t>(n) * 3 * sizeof(double));
    cl_mem dNode = ctx.alloc(cap * 16), dNeigh = ctx.alloc(cap * 16), dOwner = ctx.alloc(cap * 4),
           dPick = ctx.alloc(cap * 4), dOwnS = ctx.alloc(cap * 4);
    cl_mem dLoc = ctx.alloc(static_cast<size_t>(n) * 4), dState = ctx.alloc(n), dOvf = ctx.alloc(n);
    cl_mem dCand = ctx.alloc(static_cast<size_t>(maxcand) * 4), dCavT = ctx.alloc(static_cast<size_t>(maxcand) * MAXC * 4),
           dCavB = ctx.alloc(static_cast<size_t>(maxcand) * MAXB * 4), dCnt = ctx.alloc(static_cast<size_t>(maxcand) * 8),
           dWin = ctx.alloc(static_cast<size_t>(maxcand) * 4), dCst = ctx.alloc(maxcand), dStats = ctx.alloc(8 * 4);
    auto release_all = [&]() {
        for (cl_mem b : { dX, dNode, dNeigh, dOwner, dOwnS, dCst, dPick, dLoc, dState, dOvf, dCand, dCavT, dCavB, dCnt, dWin, dStats }) {
            clReleaseMemObject(b);
        }
    };

    ctx.write(dX, X, static_cast<size_t>(n) * 3 * sizeof(double));
    ctx.write(dNode, tm.tet_node.data(), static_cast<size_t>(nt_init) * 16);
    ctx.write(dNeigh, neigh32.data(), static_cast<size_t>(nt_init) * 16);
    ctx.write(dState, state.data(), n);
    fill_u32(ctx, dOwner, NONE, 0, cap);
    fill_u32(ctx, dOwnS, NONE, 0, cap);
    fill_u32(ctx, dPick, NONE, 0, cap);
    fill_u32(ctx, dLoc, 0, 0, n);
    {
        const uint8_t z = 0;
        cl_check(clEnqueueFillBuffer(q, dOvf, &z, 1, 0, n, 0, nullptr, nullptr), "fill");
    }

    // stats: [0] pending [1] candidates [2] tets [3] deferred [4] overflows [5] full
    //        [6] winners [7] dead slots
    uint32_t S[8] = { 0, 0, nt_init, 0, 0, 0, 0, 0 };
    const uint32_t n_pre = 4 + static_cast<uint32_t>(st.sampled);
    uint32_t pending = n - n_pre, stall = 0;
    const cl_uint un = n, umc = maxcand;
    static const bool profile = std::getenv("TN_GDEL_PROFILE") != nullptr;
    double pms[9] = { 0 };
    clk::time_point tp = clk::now();
    auto prof = [&](int k) {
        if (profile) {
            ctx.finish();
            pms[k] += since(tp);
            tp = clk::now();
        }
    };
    static const cl_uint ufilter = std::getenv("TN_GDEL_FILTER") ? std::atoi(std::getenv("TN_GDEL_FILTER")) : 3;
    static const size_t lsz = std::getenv("TN_GDEL_LS") ? std::atoi(std::getenv("TN_GDEL_LS")) : 64;
    static const int passes = std::getenv("TN_GDEL_PASSES") ? std::atoi(std::getenv("TN_GDEL_PASSES")) : 4;

    while (pending > 0) {
        const uint32_t ntet0 = S[2];
        S[0] = S[1] = S[4] = S[5] = S[6] = 0;
        ctx.write(dStats, S, sizeof S);
        const cl_uint ucap = static_cast<cl_uint>(cap);
        P.locate.a(dX).a(dNode).a(dNeigh).a(dLoc).a(dState).a(dPick).a(dStats).a(un).run(q, n); prof(0);
        P.pick.a(dLoc).a(dState).a(dPick).a(dNeigh).a(dCand).a(dStats).a(un).a(umc).a(ufilter).run(q, n);
        P.unpick.a(dLoc).a(dState).a(dPick).a(un).run(q, n); prof(1);
        P.cavity.a(dX).a(dNode).a(dNeigh).a(dLoc).a(dState).a(dOvf).a(dCand).a(dCavT).a(dCavB).a(dCnt).a(dCst)
                .a(dStats).a(umc).run(q, maxcand, lsz); prof(2);

        for (int pass = 0; pass < passes; ++pass) {
            P.claim.a(dCand).a(dCavT).a(dCavB).a(dCnt).a(dCst).a(dOwner).a(dOwnS).a(dStats).a(umc).run(q, maxcand); prof(3);
            P.check.a(dCand).a(dCavT).a(dCavB).a(dCnt).a(dCst).a(dOwner).a(dOwnS).a(dStats).a(umc).run(q, maxcand); prof(4);
            P.fix.a(dCavT).a(dCavB).a(dCnt).a(dCst).a(dOwner).a(dOwnS).a(dWin).a(dStats).a(umc).a(ucap)
                    .run(q, maxcand); prof(5);
            if (pass + 1 < passes) {   // (after the last pass d_reset releases everything)
                P.clear.a(dCavT).a(dCavB).a(dCnt).a(dCst).a(dOwner).a(dOwnS).a(dStats).a(umc).run(q, maxcand);
                prof(6);
            }
        }

        P.commit.a(dCand).a(dCavT).a(dCavB).a(dCnt).a(dCst).a(dWin).a(dNode).a(dNeigh).a(dState).a(dStats).a(umc)
                .run(q, maxcand, 64); prof(7);
        P.reset.a(dCavT).a(dCavB).a(dCnt).a(dOwner).a(dOwnS).a(dStats).a(umc).run(q, maxcand); prof(8);
        ctx.read(dStats, S, sizeof S);
        ++st.rounds;

        if (S[5]) {   // out of tet slots: grow, and redo the round (nothing was committed)
            const size_t nc = cap + cap / 3 + 4096;
            dNode = grow(ctx, dNode, cap * 4, nc * 4, 0, false);
            dNeigh = grow(ctx, dNeigh, cap * 4, nc * 4, 0, false);
            dOwner = grow(ctx, dOwner, cap, nc, NONE, true);
            dPick = grow(ctx, dPick, cap, nc, NONE, true);
            dOwnS = grow(ctx, dOwnS, cap, nc, NONE, true);
            cap = nc;
            S[2] = ntet0;
            continue;
        }

        st.gpu_inserted += S[6];   // (S[3], the deferred points, is cumulative)
        pending = n - n_pre - static_cast<uint32_t>(st.gpu_inserted) - S[3];

        if (verbose) {
            std::fprintf(stderr, "[gdel] round %3d: pending %u cand %u win %u ovf %u tets %u deferred %u\n", st.rounds,
                         S[0], S[1], S[6], S[4], S[2], S[3]);
        }

        stall = S[6] ? 0 : stall + 1;

        // the tail (a few hundred points in long, thin rounds) or no progress: the
        // exact CPU insertion does the rest faster
        if (pending <= std::max<uint32_t>(256, n / 1000) || stall >= 8 || st.rounds > 4000) {
            break;
        }
    }

    const size_t nt = S[2];
    ctx.finish();

    if (profile) {
        std::fprintf(stderr, "[gdel] ms: locate %.0f pick %.0f cavity %.0f claim %.0f check %.0f fix %.0f clear %.0f "
                     "commit %.0f reset %.0f\n", pms[0], pms[1], pms[2], pms[3], pms[4], pms[5], pms[6], pms[7], pms[8]);
    }

    st.ms_gpu = since(t0);
    t0 = clk::now();

    std::vector<uint32_t> node(nt * 4), neigh(nt * 4), loc(n);
    ctx.read(dNode, node.data(), nt * 16);
    ctx.read(dNeigh, neigh.data(), nt * 16);
    ctx.read(dState, state.data(), n);
    ctx.read(dLoc, loc.data(), static_cast<size_t>(n) * 4);
    release_all();

    tm.tet_node.assign(node.begin(), node.end());
    tm.tet_neigh.assign(neigh.begin(), neigh.end());
    tm.mark_tetrahedra.assign(nt, 0);

    for (size_t t = 0; t < nt; ++t) {
        if (node[4 * t] == DEAD) {
            tm.pushAndMarkDeletedTets(static_cast<uint64_t>(t) << 2);
            ++st.dead;
        } else if (node[4 * t + 3] != NONE) {
            for (int c = 0; c < 4; ++c) {
                tm.inc_tet[node[4 * t + c]] = t;
            }
        }
    }

    st.ms_load = since(t0);
    t0 = clk::now();

    // the rest (deferred or stalled points) by the exact CPU insertion, walking
    // from where the device located them
    for (uint32_t v = 0; v < n; ++v) {
        if (state[v] == 1) {
            continue;
        }

        uint64_t t = loc[v];

        while (t < nt && node[4 * t] == DEAD) {
            t = neigh[4 * t] >> 2;
        }

        uint64_t ct = t < nt ? (t << 2) : 0;

        while (tm.isToDelete(ct)) {   // any live tet will do as the walk's start
            ct = (ct + 4) % (static_cast<uint64_t>(tm.numTets()) << 2);
        }

        tm.insertExistingVertex(v, ct);
        ++st.deferred;
    }

    tm.removeDelTets();
    st.ms_cpu = since(t0);

    if (std::getenv("TN_GDEL_CHECK")) {
        ::TetMesh ref;
        ref.init_vertices(X, n);
        ref.tetrahedrize();
        auto canon = [](const ::TetMesh& m) {
            std::vector<std::array<uint32_t, 4>> T(m.numTets());

            for (uint32_t t = 0; t < m.numTets(); ++t) {
                std::array<uint32_t, 4> a = { m.tet_node[4 * t], m.tet_node[4 * t + 1], m.tet_node[4 * t + 2],
                                              m.tet_node[4 * t + 3] };
                std::sort(a.begin(), a.end());
                T[t] = a;
            }

            std::sort(T.begin(), T.end());
            return T;
        };
        const auto A = canon(tm), B = canon(ref);
        size_t bad_adj = 0;

        for (size_t c = 0; c < tm.tet_neigh.size(); ++c) {
            if (tm.tet_neigh[tm.tet_neigh[c]] != c) {
                ++bad_adj;
            }
        }

        std::fprintf(stderr, "[gdel] check: %zu tets (CPU %zu) -> %s; adjacency errors %zu\n", A.size(), B.size(),
                     A == B ? "IDENTICAL" : "DIFFERENT", bad_adj);
    }

    return true;
}

}  // namespace tn
