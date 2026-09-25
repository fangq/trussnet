// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_gpu.cpp -- see tn_gpu.h. The round structure matches relax_cpu: rebuild
// (keys -> count -> scan -> scatter -> K-nearest), then per iteration force ->
// move -> stats (histogram, max, Verlet count + local list refresh), one small
// read-back per iteration for the convergence / rebuild decisions.

#include "tn_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tn_cl_host.h"
#include "tn_log.h"

namespace tn {

namespace gpu_host {
using std::floor;
typedef uint8_t uchar;
typedef uint16_t ushort;
#define TN_G
#include "opencl/tn_grid_body.cl"
#include "opencl/tn_seed_body.cl"
#include "opencl/tn_particle_body.cl"
#undef TN_G
}  // namespace gpu_host

using gpu_host::TnHash;

namespace {

typedef std::chrono::steady_clock clk;
double since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);

    if (!f) {
        throw std::runtime_error("cannot read OpenCL source " + path);
    }

    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string program_source() {
#ifdef TN_SRC_DIR
    const std::string dir = std::string(TN_SRC_DIR) + "/opencl/";
#else
    const std::string dir = "src/opencl/";
#endif
    return "#define TN_G __global\n" + slurp(dir + "tn_grid_body.cl") + slurp(dir + "tn_seed_body.cl") +
           slurp(dir + "tn_particle_body.cl") + slurp(dir + "tn_kernels.cl");
}

// kernel launcher: set args in order, enqueue a 1-D range
struct Kern {
    cl_kernel k = nullptr;
    int na = 0;
    template <typename T>
    Kern& a(const T& v) {
        cl_check(clSetKernelArg(k, na++, sizeof(T), &v), "clSetKernelArg");
        return *this;
    }
    Kern& local(size_t bytes) {
        cl_check(clSetKernelArg(k, na++, bytes, nullptr), "clSetKernelArg(local)");
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

}  // namespace

void relax_cl(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st, int device) {
    ClCtx ctx;
    ctx.init(device);
    const clk::time_point tb = clk::now();
    std::string opts = "-cl-fp32-correctly-rounded-divide-sqrt -cl-std=CL1.2";

    if (const char* e = std::getenv("TN_CL_OPTS")) {
        opts += std::string(" ") + e;
    }

    cl_program prog = ctx.build(program_source(), opts);
    const double ms_build = since(tb);
    cl_int err = CL_SUCCESS;
    auto mk = [&](const char* name) {
        Kern K;
        K.k = clCreateKernel(prog, name, &err);
        cl_check(err, name);
        return K;
    };
    Kern kKeys = mk("k_keys"), kScat = mk("k_scatter"), kNbr = mk("k_neighbors"), kForce = mk("k_force"),
         kMove = mk("k_move"), kStats = mk("k_stats"), kScanB = mk("k_scan_block"), kScanA = mk("k_scan_add");
    cl_command_queue q = ctx.queue();
    const int n = static_cast<int>(nd.size());

    // hash geometry: identical to relax_cpu
    TnHash H;
    std::memset(&H, 0, sizeof(H));
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

    // buffers
    const size_t nv = static_cast<size_t>(g.nx) * g.ny * g.nz;
    auto up = [&](const void* p, size_t bytes, cl_mem_flags f = CL_MEM_READ_ONLY) {
        cl_mem m = ctx.alloc(bytes, f);

        if (bytes) {
            ctx.write(m, p, bytes);
        }

        return m;
    };
    cl_mem dL = up(g.L->data(), nv * 2), dCnt = up(g.bl_cnt.data(), g.bl_cnt.size() * 4),
           dLab = up(g.bl_lab.data(), g.bl_lab.size() * 2), dSlot = up(g.bl_slot.data(), g.bl_slot.size() * 4),
           dPhi = up(g.phi.data(), g.phi.size() * 4), dH = up(g.h.data(), nv * 4);
    cl_mem dP = up(nd.P.data(), nd.P.size() * 4, CL_MEM_READ_WRITE),
           dP0 = up(nd.P.data(), nd.P.size() * 4, CL_MEM_READ_WRITE),
           dNl = up(nd.lab.data(), nd.lab.size() * 2), dTyp = up(nd.typ.data(), nd.typ.size(), CL_MEM_READ_WRITE),
           dPart = up(nd.part.data(), nd.part.size() * 2, CL_MEM_READ_WRITE);
    cl_mem dKey = ctx.alloc(static_cast<size_t>(n) * 4), dBin = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4),
           dStart = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4), dCur = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4),
           dSorted = ctx.alloc(static_cast<size_t>(n) * 4), dNbr = ctx.alloc(static_cast<size_t>(n) * TN_K * 4),
           dNnb = ctx.alloc(static_cast<size_t>(n) * 4), dF = ctx.alloc(static_cast<size_t>(n) * 16),
           dMv = ctx.alloc(static_cast<size_t>(n) * 4), dStats = ctx.alloc(34 * 4),
           dHn = ctx.alloc(static_cast<size_t>(n) * 4);
    // scan scratch: block sums per level
    std::vector<cl_mem> sums;
    std::vector<int> sums_n;
    {
        int m = nkeys + 1;

        while (true) {
            const int nb = (m + 511) / 512;
            sums.push_back(ctx.alloc(static_cast<size_t>(nb) * 4));
            sums_n.push_back(nb);

            if (nb <= 1) {
                break;
            }

            m = nb;
        }
    }
    // exclusive scan of `buf` (length m) into `out`, in place over the levels
    std::function<void(cl_mem, cl_mem, int, int)> scan = [&](cl_mem in, cl_mem out, int m, int lev) {
        const int nb = (m + 511) / 512;
        kScanB.a(in).a(out).a(m).a(sums[lev]).run(q, static_cast<size_t>(nb) * 256, 256);

        if (nb > 1) {
            scan(sums[lev], sums[lev], nb, lev + 1);
            kScanA.a(out).a(m).a(sums[lev]).run(q, static_cast<size_t>(m), 256);
        }
    };
    const int nx = g.nx, ny = g.ny, nz = g.nz, nbx = g.nbx, nby = g.nby, nbz = g.nbz;
    const float vx = g.vs[0], vy = g.vs[1], vz = g.vs[2];
    const float t = prm.t, skin = prm.skin;
    auto dims = [&](Kern& K) -> Kern& {
        return K.a(nx).a(ny).a(nz).a(nbx).a(nby).a(nbz).a(vx).a(vy).a(vz);
    };
    const int zero = 0;

    const bool prof = std::getenv("TN_CL_PROFILE") != nullptr;
    double pk[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };   // sort, neighbours, -, move, stats, read
    auto tick = [&](int slot, clk::time_point& tt) {
        if (prof) {
            ctx.finish();
            pk[slot] += since(tt);
            tt = clk::now();
        }
    };
    auto rebuild = [&]() {
        clk::time_point t0 = clk::now();
        cl_check(clEnqueueFillBuffer(q, dBin, &zero, 4, 0, (static_cast<size_t>(nkeys) + 1) * 4, 0, nullptr, nullptr),
                 "fill");
        dims(kKeys.a(H)).a(dH).a(dP).a(n).a(t).a(skin).a(dKey).a(dBin).a(dHn).run(q, n);
        scan(dBin, dStart, nkeys + 1, 0);
        cl_check(clEnqueueCopyBuffer(q, dStart, dCur, 0, 0, (static_cast<size_t>(nkeys) + 1) * 4, 0, nullptr, nullptr),
                 "copy");
        kScat.a(dKey).a(n).a(dCur).a(dSorted).run(q, n);
        clk::time_point tq = t0;
        tick(0, tq);
        kNbr.a(H).a(dHn).a(dP).a(dNl).a(dTyp).a(dStart).a(dSorted).a(t).a(skin).a(n).a(dNbr).a(dNnb).run(q, n, 64);
        tick(1, tq);
        cl_check(clEnqueueCopyBuffer(q, dP, dP0, 0, 0, static_cast<size_t>(n) * 12, 0, nullptr, nullptr), "copy");
        ctx.finish();
        ++st.rebuilds;
        st.ms_hash += since(t0);
    };

    rebuild();
    const int voxmode = prm.voxel_trap ? 1 : 0;
    int stats[34];

    for (int it = 0; it < prm.max_iters; ++it) {
        clk::time_point t0 = clk::now();
        kForce.a(dHn).a(dP).a(dTyp).a(dNbr).a(dNnb).a(prm.fscale).a(prm.fsurf).a(n).a(dF).run(q, n);
        ctx.finish();
        st.ms_force += since(t0);
        clk::time_point t1 = clk::now(), tp = clk::now();
        dims(kMove.a(dL).a(dCnt).a(dLab).a(dSlot).a(dPhi)).a(dH).a(dF).a(prm.dt).a(prm.maxstep).a(prm.snap).a(voxmode)
            .a(n).a(dP).a(dNl).a(dTyp).a(dPart).a(dMv).a(dHn).run(q, n, 64);
        tick(3, tp);
        cl_check(clEnqueueFillBuffer(q, dStats, &zero, 4, 0, 34 * 4, 0, nullptr, nullptr), "fill");
        kStats.a(H).a(dHn).a(dP).a(dP0).a(dMv).a(dNl).a(dTyp).a(dStart).a(dSorted).a(t).a(skin).a(n).a(dNbr)
            .a(dNnb).a(dStats).run(q, n, 128);
        tick(4, tp);
        ctx.read(dStats, stats, sizeof(stats));
        tick(5, tp);
        st.ms_move += since(t1);
        st.iters = it + 1;
        float mmax;
        std::memcpy(&mmax, &stats[32], 4);
        st.last_move = mmax;
        int acc = 0, b99 = 31;

        for (int b = 0; b < 32; ++b) {
            acc += stats[b];

            if (acc >= 0.99 * n) {
                b99 = b;
                break;
            }
        }

        const float p99 = std::ldexp(1.0f, b99 - 20 + 1);
        st.last_p99 = p99;

        if (prm.verbose && (it % 50 == 0)) {
            TN_FPRINTF(stderr, "[relax] iter %d: max move %.4g h, p99 < %.3g h\n", it, mmax, p99);
        }

        if (p99 < prm.dptol) {
            break;
        }

        if (stats[33] > n / 1000) {
            rebuild();
        }
    }

    ctx.read(dP, nd.P.data(), nd.P.size() * 4);
    ctx.read(dTyp, nd.typ.data(), nd.typ.size());
    ctx.read(dPart, nd.part.data(), nd.part.size() * 2);
    st.n_interior = st.n_interface = st.n_junction = st.n_corner = 0;

    for (uint8_t ty : nd.typ) {
        st.n_interior += ty == TN_INTERIOR;
        st.n_interface += ty == TN_INTERFACE;
        st.n_junction += ty == TN_JUNCTION;
        st.n_corner += ty == TN_CORNER;
    }

    if (prm.verbose) {
        TN_FPRINTF(stderr, "[relax] OpenCL %s: program build %.0f ms\n", ctx.deviceName().c_str(), ms_build);
    }

    if (prof) {
        TN_FPRINTF(stderr, "[clprof] hash sort %.0f, neighbours %.0f, move %.0f, stats %.0f, read %.0f ms\n", pk[0],
                   pk[1], pk[3], pk[4], pk[5]);
    }

    for (cl_mem m : { dL, dCnt, dLab, dSlot, dPhi, dH, dP, dP0, dNl, dTyp, dPart, dKey, dBin, dStart, dCur, dSorted,
                      dNbr, dNnb, dF, dMv, dStats, dHn }) {
        clReleaseMemObject(m);
    }

    for (cl_mem m : sums) {
        clReleaseMemObject(m);
    }

    for (Kern* K : { &kKeys, &kScat, &kNbr, &kForce, &kMove, &kStats, &kScanB, &kScanA }) {
        clReleaseKernel(K->k);
    }

    clReleaseProgram(prog);
}

}  // namespace tn
