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
#include "tn_cl_sources.h"
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

std::string joined(const char* const* parts) {
    std::string s;

    for (; *parts; ++parts) {
        s += *parts;
    }

    return s;
}

// the kernels are embedded (tn_cl_sources.h); TN_CL_DIR=<dir> reads them from
// there instead (kernel development without a rebuild)
std::string cl_file(const char* name, const char* const* embedded) {
    if (const char* d = std::getenv("TN_CL_DIR")) {
        return slurp(std::string(d) + "/" + name);
    }

    return joined(embedded);
}

std::string program_source() {
    return "#define TN_G __global\n" + cl_file("tn_grid_body.cl", tn_grid_body_cl) +
           cl_file("tn_seed_body.cl", tn_seed_body_cl) + cl_file("tn_particle_body.cl", tn_particle_body_cl) +
           cl_file("tn_kernels.cl", tn_kernels_cl);
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
    // Verlet trigger: rebuild once more than 1/rebuild_div of the nodes moved skin/2
    // since the build (nodes past a full skin refresh their own list anyway)
    static const int rebuild_div = std::getenv("TN_REBUILD_DIV") ? std::atoi(std::getenv("TN_REBUILD_DIV")) : 300;
    const clk::time_point ti = clk::now();
    ClCtx& ctx = cl_shared_ctx(device);
    const double ms_init = since(ti);
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
    Kern kGather = mk("k_gather");
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
    const clk::time_point tu = clk::now();
    const size_t nv = static_cast<size_t>(g.nx) * g.ny * g.nz;
    auto upf = [&](const void* p, size_t bytes, cl_mem_flags f) {
        cl_mem m = ctx.alloc(bytes, f);

        if (bytes) {
            ctx.write(m, p, bytes);
        }

        return m;
    };
    auto up = [&](const void* p, size_t bytes) {
        return upf(p, bytes, CL_MEM_READ_ONLY);
    };
    cl_mem dL = up(g.L->data(), nv * 2), dCnt = up(g.bl_cnt.data(), g.bl_cnt.size() * 4),
           dLab = up(g.bl_lab.data(), g.bl_lab.size() * 2), dSlot = up(g.bl_slot.data(), g.bl_slot.size() * 4),
           dPhi = up(g.phi.data(), g.phi.size() * 4), dH = up(g.h.data(), nv * 4);
    const float gdummy = 0.0f;
    cl_mem dGI = g.gm > 0 ? up(g.gI, nv * 4) : up(&gdummy, 4),
           dGTW = up(g.gTW.data(), std::max<size_t>(1, g.gTW.size()) * 4);
    const int gm = g.gm;
    cl_mem dP = upf(nd.P.data(), nd.P.size() * 4, CL_MEM_READ_WRITE),
           dP0 = upf(nd.P.data(), nd.P.size() * 4, CL_MEM_READ_WRITE),
           dNl = up(nd.lab.data(), nd.lab.size() * 2), dTyp = upf(nd.typ.data(), nd.typ.size(), CL_MEM_READ_WRITE),
           dPart = upf(nd.part.data(), nd.part.size() * 2, CL_MEM_READ_WRITE);
    cl_mem dKey = ctx.alloc(static_cast<size_t>(n) * 4), dBin = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4),
           dStart = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4), dCur = ctx.alloc((static_cast<size_t>(nkeys) + 1) * 4),
           dSorted = ctx.alloc(static_cast<size_t>(n) * 4), dNbr = ctx.alloc(static_cast<size_t>(n) * TN_K * 4),
           dNnb = ctx.alloc(static_cast<size_t>(n) * 4), dF = ctx.alloc(static_cast<size_t>(n) * 16),
           dMv = ctx.alloc(static_cast<size_t>(n) * 4), dStats = ctx.alloc(34 * 4),
           dHn = ctx.alloc(static_cast<size_t>(n) * 4), dPs = ctx.alloc(static_cast<size_t>(n) * 16);
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
        kGather.a(dSorted).a(dP).a(dHn).a(n).a(dPs).run(q, n);
        kNbr.a(H).a(dHn).a(dP).a(dNl).a(dTyp).a(dStart).a(dSorted).a(dPs).a(t).a(skin).a(n).a(dNbr).a(dNnb).run(q, n, 64);
        tick(1, tq);
        cl_check(clEnqueueCopyBuffer(q, dP, dP0, 0, 0, static_cast<size_t>(n) * 12, 0, nullptr, nullptr), "copy");
        ctx.finish();
        ++st.rebuilds;
        st.ms_hash += since(t0);
    };

    ctx.finish();
    const double ms_upload = since(tu);
    size_t up_bytes = nv * 2 + g.bl_cnt.size() * 4 + g.bl_lab.size() * 2 + g.bl_slot.size() * 4 + g.phi.size() * 4 + nv * 4 +
                      nd.P.size() * 8 + nd.lab.size() * 2 + nd.typ.size() + nd.part.size() * 2;
    up_bytes += g.gm > 0 ? nv * 4 : 0;
    const clk::time_point tl = clk::now();
    rebuild();
    const int voxmode = prm.voxel_trap ? 1 : 0;
    int stats[34];
    // TN_RELAX_TRACE=1: where does the motion go? positions snapshotted at a few
    // iterations; at the end the net displacement since each, by node type and by
    // truss-graph distance (rings) from the nearest non-interior node
    static const bool trace = std::getenv("TN_RELAX_TRACE") != nullptr;
    const int snap_it[5] = { 50, 100, 200, 300, 400 };
    std::vector<std::vector<float>> snap(5);

    for (int it = 0; it < prm.max_iters; ++it) {
        if (trace) {
            for (int k = 0; k < 5; ++k)
                if (it == snap_it[k]) {
                    snap[k].resize(static_cast<size_t>(n) * 3);
                    ctx.read(dP, snap[k].data(), snap[k].size() * 4);
                }
        }

        clk::time_point t0 = clk::now();
        kForce.a(dHn).a(dP).a(dTyp).a(dNbr).a(dNnb).a(prm.fscale).a(prm.fsurf).a(n).a(dF).run(q, n);
        ctx.finish();
        st.ms_force += since(t0);
        clk::time_point t1 = clk::now(), tp = clk::now();
        dims(kMove.a(dL).a(dCnt).a(dLab).a(dSlot).a(dPhi).a(dGI).a(dGTW).a(gm)).a(dH).a(dF).a(prm.dt).a(prm.maxstep).a(prm.snap).a(voxmode)
            .a(n).a(dP).a(dNl).a(dTyp).a(dPart).a(dMv).a(dHn).run(q, n, 64);
        tick(3, tp);
        cl_check(clEnqueueFillBuffer(q, dStats, &zero, 4, 0, 34 * 4, 0, nullptr, nullptr), "fill");
        kStats.a(H).a(dHn).a(dP).a(dP0).a(dMv).a(dNl).a(dTyp).a(dStart).a(dSorted).a(dPs).a(t).a(skin).a(n).a(dNbr)
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

        // loose trigger for the bulk of the relaxation, strict (0.1%) near the end so
        // the final positions are relaxed with exact lists (the loose one alone cost
        // a few conforming faces on the wedge / T-junction phantoms)
        const bool endgame = it >= prm.max_iters * 4 / 5 || p99 < 4.0f * prm.dptol;

        if (stats[33] > n / (endgame ? std::max(rebuild_div, 1000) : rebuild_div)) {
            rebuild();
        }
    }

    const double ms_loop = since(tl);

    if (trace) {
        std::vector<float> Pf(static_cast<size_t>(n) * 3), hn(n), mv(n);
        std::vector<uint8_t> ty(n);
        std::vector<int> nb(static_cast<size_t>(n) * TN_K), nn(n);
        ctx.read(dP, Pf.data(), Pf.size() * 4);
        ctx.read(dHn, hn.data(), hn.size() * 4);
        ctx.read(dMv, mv.data(), mv.size() * 4);
        ctx.read(dTyp, ty.data(), ty.size());
        ctx.read(dNbr, nb.data(), nb.size() * 4);
        ctx.read(dNnb, nn.data(), nn.size() * 4);
        // rings: BFS over the truss from the interface / junction / corner nodes
        std::vector<int> ring(n, 99), q;

        for (int i = 0; i < n; ++i)
            if (ty[i] != TN_INTERIOR) {
                ring[i] = 0;
                q.push_back(i);
            }

        for (size_t h = 0; h < q.size(); ++h) {
            const int i = q[h];

            for (int k = 0; k < nn[i] && k < TN_K; ++k) {
                const int j = nb[static_cast<size_t>(i) * TN_K + k];

                if (j >= 0 && j < n && ring[j] > ring[i] + 1) {
                    ring[j] = ring[i] + 1;
                    q.push_back(j);
                }
            }
        }

        const char* rn[8] = { "iface", "ring1", "ring2", "ring3", "ring4", "ring5-8", "ring9+", "all" };
        auto bucket = [&](int r) {
            return r == 0 ? 0 : r <= 4 ? r : r <= 8 ? 5 : 6;
        };
        std::vector<size_t> cnt(8, 0);

        for (int i = 0; i < n; ++i) {
            ++cnt[bucket(ring[i])];
        }

        cnt[7] = n;
        TN_FPRINTF(stderr, "[trace] %d iterations; nodes per ring:", st.iters);

        for (int b = 0; b < 8; ++b) {
            TN_FPRINTF(stderr, " %s %zu", rn[b], cnt[b]);
        }

        TN_FPRINTF(stderr, "\n[trace] fraction of nodes with net |dp| > 0.02 h / 0.1 h since iteration X, by ring:\n");

        for (int k = 0; k < 5; ++k) {
            if (snap[k].empty()) {
                continue;
            }

            std::vector<size_t> a2(8, 0), a10(8, 0);

            for (int i = 0; i < n; ++i) {
                const float dx = Pf[3 * i] - snap[k][3 * i], dy = Pf[3 * i + 1] - snap[k][3 * i + 1],
                            dz = Pf[3 * i + 2] - snap[k][3 * i + 2];
                const float d = std::sqrt(dx * dx + dy * dy + dz * dz) / hn[i];
                const int b = bucket(ring[i]);

                for (int bb : { b, 7 }) {
                    a2[bb] += d > 0.02f;
                    a10[bb] += d > 0.1f;
                }
            }

            TN_FPRINTF(stderr, "[trace]   since %3d:", snap_it[k]);

            for (int b = 0; b < 8; ++b)
                TN_FPRINTF(stderr, " %s %.3f/%.3f", rn[b], cnt[b] ? double(a2[b]) / cnt[b] : 0.0,
                           cnt[b] ? double(a10[b]) / cnt[b] : 0.0);

            TN_FPRINTF(stderr, "\n");
        }

        // oscillators: a large last step but little net motion over the last 100 iterations
        if (!snap[4].empty()) {
            size_t big = 0, osc = 0;
            std::vector<size_t> byty(4, 0);

            for (int i = 0; i < n; ++i) {
                if (mv[i] <= 0.05f) {
                    continue;
                }

                ++big;
                const float dx = Pf[3 * i] - snap[4][3 * i], dy = Pf[3 * i + 1] - snap[4][3 * i + 1],
                            dz = Pf[3 * i + 2] - snap[4][3 * i + 2];

                if (std::sqrt(dx * dx + dy * dy + dz * dz) / hn[i] < 0.1f) {
                    ++osc;
                    ++byty[std::min<int>(ty[i], 3)];
                }
            }

            TN_FPRINTF(stderr, "[trace] last step > 0.05 h: %zu nodes, of which %zu oscillate (net < 0.1 h since 400): "
                       "interior %zu interface %zu junction %zu corner %zu\n", big, osc, byty[0], byty[1], byty[2], byty[3]);
        }
    }
    const clk::time_point tr = clk::now();
    ctx.read(dP, nd.P.data(), nd.P.size() * 4);
    ctx.read(dTyp, nd.typ.data(), nd.typ.size());
    ctx.read(dPart, nd.part.data(), nd.part.size() * 2);
    const double ms_read = since(tr);
    st.n_interior = st.n_interface = st.n_junction = st.n_corner = 0;

    for (uint8_t ty : nd.typ) {
        st.n_interior += ty == TN_INTERIOR;
        st.n_interface += ty == TN_INTERFACE;
        st.n_junction += ty == TN_JUNCTION;
        st.n_corner += ty == TN_CORNER;
    }

    if (prm.verbose) {
        TN_FPRINTF(stderr, "[relax] OpenCL %s: context %.0f ms, program build %.0f ms, upload %.1f MB in %.0f ms, "
                   "iterations %.0f ms, read-back %.0f ms\n", ctx.deviceName().c_str(), ms_init, ms_build, up_bytes / 1e6,
                   ms_upload, ms_loop, ms_read);
    }

    if (prof) {
        TN_FPRINTF(stderr, "[clprof] hash sort %.0f, neighbours %.0f, move %.0f, stats %.0f, read %.0f ms\n", pk[0],
                   pk[1], pk[3], pk[4], pk[5]);
    }

    for (cl_mem m : { dL, dCnt, dLab, dSlot, dPhi, dH, dP, dP0, dNl, dTyp, dPart, dKey, dBin, dStart, dCur, dSorted,
                      dNbr, dNnb, dF, dMv, dStats, dHn, dGI, dGTW, dPs }) {
        clReleaseMemObject(m);
    }

    for (cl_mem m : sums) {
        clReleaseMemObject(m);
    }

    for (Kern* K : { &kGather, &kKeys, &kScat, &kNbr, &kForce, &kMove, &kStats, &kScanB, &kScanA }) {
        clReleaseKernel(K->k);
    }

    clReleaseProgram(prog);
}

}  // namespace tn
