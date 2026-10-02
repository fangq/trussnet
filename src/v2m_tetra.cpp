// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_tetra.cpp -- see v2m_tetra.h.

#include "v2m_tetra.h"
#include "v2m_opt.h"

#ifdef _OPENMP
    #include <omp.h>
#else
inline int omp_get_max_threads() {
    return 1;
}
inline int omp_get_thread_num() {
    return 0;
}
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "delaunay.h"
#include "v2m_omp.h"
#include "v2m_gdel.h"

namespace tn {

namespace tetra_host {

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
#define V2M_G
#include "opencl/v2m_grid_body.cl"
#include "opencl/v2m_sdf_body.cl"
#include "opencl/v2m_seed_body.cl"
#include "opencl/v2m_particle_body.cl"
#undef V2M_G

}  // namespace tetra_host

using namespace tetra_host;

namespace {

typedef std::chrono::steady_clock clk;
double since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

// circumcentre of tet (a,b,c,d); false if degenerate
bool circumcentre(const double* a, const double* b, const double* c, const double* d, double* o) {
    double u[3], v[3], w[3];

    for (int k = 0; k < 3; ++k) {
        u[k] = b[k] - a[k];
        v[k] = c[k] - a[k];
        w[k] = d[k] - a[k];
    }

    const double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2], vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
                 ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const double vxw[3] = { v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0] };
    const double wxu[3] = { w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0] };
    const double uxv[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
    const double det = 2.0 * (u[0] * vxw[0] + u[1] * vxw[1] + u[2] * vxw[2]);

    if (std::fabs(det) < 1e-300) {
        return false;
    }

    for (int k = 0; k < 3; ++k) {
        o[k] = a[k] + (uu * vxw[k] + vv * wxu[k] + ww * uxv[k]) / det;
    }

    return true;
}

}  // namespace

// minimum dihedral (deg) and Joe-Liu quality 12 (3V)^(2/3) / sum l^2 of a tet
void tet_quality(const double* p[4], double& mindih, double& jl, double& vol) {
    double e[6][3];
    const int ei[6][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };
    double l2 = 0.0;

    for (int k = 0; k < 6; ++k) {
        for (int c = 0; c < 3; ++c) {
            e[k][c] = p[ei[k][1]][c] - p[ei[k][0]][c];
        }

        l2 += e[k][0] * e[k][0] + e[k][1] * e[k][1] + e[k][2] * e[k][2];
    }

    const double* a = e[0];
    const double* b = e[1];
    const double* c = e[2];
    const double v6 = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                      a[2] * (b[0] * c[1] - b[1] * c[0]);
    vol = std::fabs(v6) / 6.0;
    jl = l2 > 0 ? 12.0 * std::pow(3.0 * vol, 2.0 / 3.0) / l2 : 0.0;
    // face normals, outward by orientation against the opposite vertex
    double n[4][3];

    for (int f = 0; f < 4; ++f) {
        int q[3], m = 0;

        for (int k = 0; k < 4; ++k)
            if (k != f) {
                q[m++] = k;
            }

        double u[3], w[3], s[3];

        for (int k = 0; k < 3; ++k) {
            u[k] = p[q[1]][k] - p[q[0]][k];
            w[k] = p[q[2]][k] - p[q[0]][k];
            s[k] = p[f][k] - p[q[0]][k];
        }

        double x[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0] };
        const double sg = (x[0] * s[0] + x[1] * s[1] + x[2] * s[2]) > 0 ? -1.0 : 1.0;
        const double xl = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) + 1e-300;

        for (int k = 0; k < 3; ++k) {
            n[f][k] = sg * x[k] / xl;
        }
    }

    mindih = 180.0;

    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            double cs = -(n[i][0] * n[j][0] + n[i][1] * n[j][1] + n[i][2] * n[j][2]);
            cs = cs < -1 ? -1 : (cs > 1 ? 1 : cs);
            mindih = std::min(mindih, std::acos(cs) * 57.29577951308232);
        }
}

// A repair: the crossing of edge (x, y) with the a|b interface (b = 0: the
// exterior).
struct Fix {
    uint32_t x, y;   // y == UINT32_MAX: a junction fix at the centroid of tet x's nodes;
                     // y == V2M_FACEFIX: a face fix, the centroid of face tv[0..2] onto a|b
    int a, b;
    uint32_t tv[4];
};

static const uint32_t V2M_FACEFIX = UINT32_MAX - 1;

// Unique edges (x < y) of a tet list, in parallel: node -> tet CSR (count ->
// scan -> fill), then each node x collects its neighbours y > x from its tets and
// sorts / uniques that short list. Replaces a serial sort of 6 pairs per tet.
static std::vector<std::pair<int, int>> unique_edges(const std::vector<int>& tets, int nn) {
    const int64_t nt = static_cast<int64_t>(tets.size() / 4);
    std::vector<int> cnt(static_cast<size_t>(nn) + 1, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < 4 * nt; ++i) {   // (parallel: atomic counts / slots; the
        #pragma omp atomic                   // per-node lists are sorted below anyway)
        ++cnt[tets[i] + 1];
    }

    for (int i = 0; i < nn; ++i) {
        cnt[i + 1] += cnt[i];
    }

    std::vector<int> cur(cnt.begin(), cnt.end() - 1), n2t(static_cast<size_t>(cnt[nn]));
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < 4 * nt; ++i) {
        int ps;
        #pragma omp atomic capture
        ps = cur[tets[i]]++;
        n2t[ps] = static_cast<int>(i >> 2);
    }

    // one flat buffer instead of a vector per node: node x's slice holds up to 3
    // candidates per incident tet (the other three vertices), sorted / uniqued in
    // place (allocating ~1e5 small vectors per call dominated this function)
    std::vector<int> ecnt(static_cast<size_t>(nn) + 1, 0);
    std::unique_ptr<int[]> buf(new int[static_cast<size_t>(cnt[nn]) * 3]);
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int x = 0; x < nn; ++x) {
        int* L = buf.get() + static_cast<size_t>(cnt[x]) * 3;
        int m = 0;

        for (int s = cnt[x]; s < cnt[x + 1]; ++s)
            for (int k = 0; k < 4; ++k) {
                const int y = tets[4 * static_cast<int64_t>(n2t[s]) + k];

                if (y > x) {
                    L[m++] = y;
                }
            }

        std::sort(L, L + m);
        ecnt[x + 1] = static_cast<int>(std::unique(L, L + m) - L);
    }

    std::vector<int> off(static_cast<size_t>(nn) + 1, 0);

    for (int x = 0; x < nn; ++x) {
        off[x + 1] = off[x] + ecnt[x + 1];
    }

    std::vector<std::pair<int, int>> e(static_cast<size_t>(off[nn]));
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int x = 0; x < nn; ++x) {
        const int* L = buf.get() + static_cast<size_t>(cnt[x]) * 3;

        for (int k = 0; k < ecnt[x + 1]; ++k) {
            e[off[x] + k] = std::make_pair(x, L[k]);
        }
    }

    return e;
}

// label set of node v: {a}, {a,b} interface, {a,b,c} junction, {a,b,c,d} corner
static int node_label_set(const Nodes& nd, uint32_t v, int* out) {
    out[0] = nd.lab[v];
    int n = 1;

    if (nd.typ[v] != V2M_INTERIOR) {
        out[n++] = nd.part[2 * v];
    }

    if (nd.typ[v] >= V2M_JUNCTION) {
        out[n++] = nd.part[2 * v + 1];
    }

    if (nd.typ[v] == V2M_CORNER) {
        out[n++] = nd.part3[v];
    }

    return n;
}

// the OpenCL device of the full Delaunay builds (-2 = the CPU; see set_gpu_delaunay)
static int g_del_device = -2;

void set_gpu_delaunay(int device) {
    g_del_device = device;
}

// `live` persists across repair rounds: nullptr or rebuild -> a fresh Delaunay of
// all nodes; else the nodes appended since the last round are inserted into it
// incrementally (the repairs only ADD nodes then; a round that moved nodes
// asks for a rebuild).
static void tessellate_once(const Grid& g, const Nodes& nd, bool voxel_mode, TetOut& m, TetStats& st,
                            std::unique_ptr<::TetMesh>& live, bool rebuild, std::vector<Fix>& facefixes,
                            std::vector<Fix>& fixes, std::vector<std::array<uint32_t, 5>>& span_tets,
                            std::vector<std::pair<int, int>>& eout_prev, int first_new, bool surface_only) {
    OmpThreadCap cap;
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
    const int n = static_cast<int>(nd.size());
    std::vector<double> X(nd.P.begin(), nd.P.end());

    const bool timing = std::getenv("V2M_TESS_TIMING") != nullptr;
    clk::time_point tm0 = clk::now();
    auto lap = [&](const char* what) {
        if (timing) {
            std::fprintf(stderr, "[tt] %-10s %8.0f ms\n", what, since(tm0));
        }

        tm0 = clk::now();
    };

    // 1. exact Delaunay (fresh, or incremental insertion of the new nodes)
    clk::time_point t0 = clk::now();

    if (!live || rebuild) {
        live.reset(new ::TetMesh());
        bool done = false;
#ifdef V2M_HAS_OPENCL

        if (g_del_device > -2) {
            GdelStats gs;
            done = gdel_tetrahedrize(X.data(), static_cast<uint32_t>(n), *live, gs, g_del_device);

            if (done && timing) {
                std::fprintf(stderr, "[tt] gdel: %zu sampled, %d rounds, %zu on device, %zu deferred, %zu dead; build %.0f gpu %.0f "
                             "load %.0f cpu %.0f ms\n", gs.sampled, gs.rounds, gs.gpu_inserted, gs.deferred, gs.dead, gs.ms_build,
                             gs.ms_gpu, gs.ms_load, gs.ms_cpu);
            }

            if (!done) {
                live.reset(new ::TetMesh());
            }
        }

#endif

        if (!done) {
            live->init_vertices(X.data(), static_cast<uint32_t>(n));
            live->tetrahedrize();
        }

        canonicalize_tets(*live);   // same mesh from either path, in every build, run to run
    } else if (live->numVertices() < static_cast<uint32_t>(n)) {
        uint64_t ct = 0;

        for (uint32_t v = live->numVertices(); v < static_cast<uint32_t>(n); ++v) {
            live->pushVertex(new explicitPoint(X[3 * v], X[3 * v + 1], X[3 * v + 2]));
            live->insertExistingVertex(v, ct);
        }

        live->removeDelTets();
    }

    ::TetMesh& tin = *live;
    st.ms_delaunay = since(t0);
    lap("delaunay");
    const int64_t nt = tin.numTets();

    // 2. tet labels from the NODES' label sets: {a} for an interior node, {a,b}
    // on an interface, {a,b,c} on a junction. A tet takes the label common to all
    // four sets (the centroid breaks ties among several); a face shared by tets
    // labelled x != y then has nodes carrying both labels, i.e. lies on the x|y
    // interface BY CONSTRUCTION. An empty intersection is a genuinely crossing
    // (spanning) tet: labelled by its centroid and counted as a violation.
    // (V2M_CIRCUMCENTRE_LABEL: the plain restricted-Delaunay rule, for comparison;
    // on the relaxed nodes it dropped 1% of a convex ball's volume -- flat surface
    // tets whose circumcentre falls just outside.)
    const bool use_cc = std::getenv("V2M_CIRCUMCENTRE_LABEL") != nullptr;
    clk::time_point t1 = clk::now();
    std::vector<int> tl(static_cast<size_t>(nt), -1);   // -1 ghost
    std::vector<char> conflict(static_cast<size_t>(nt), 0);
    auto node_labels = [&](uint32_t v, int* out) {
        return node_label_set(nd, v, out);
    };
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t t = 0; t < nt; ++t) {
        if (tin.isGhost(static_cast<uint64_t>(t))) {
            continue;
        }

        const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
        double o[3];
        const double* a = &X[3 * v[0]];
        const double* b = &X[3 * v[1]];
        const double* c = &X[3 * v[2]];
        const double* e = &X[3 * v[3]];

        if (!use_cc || !circumcentre(a, b, c, e, o)) {
            for (int k = 0; k < 3; ++k) {
                o[k] = 0.25 * (a[k] + b[k] + c[k] + e[k]);
            }
        }

        int sec;
        float mg;
        const int lc = v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm,
                                   voxel_mode ? 1 : 0, static_cast<float>(o[0]), static_cast<float>(o[1]),
                                   static_cast<float>(o[2]), &sec, &mg);

        if (use_cc) {
            tl[t] = lc;
            continue;
        }

        // intersection of the four label sets
        int I[4];
        int ni = node_labels(v[0], I);

        for (int k = 1; k < 4 && ni > 0; ++k) {
            int S[4];
            const int ns = node_labels(v[k], S);
            int m = 0;

            for (int x = 0; x < ni; ++x)
                for (int y = 0; y < ns; ++y)
                    if (I[x] == S[y]) {
                        I[m++] = I[x];
                        break;
                    }

            ni = m;
        }

        if (ni == 0) {
            conflict[t] = 1;
            tl[t] = lc;
        } else if (ni == 1) {
            tl[t] = I[0];
        } else if (!g.feat.empty()) {
            // several, shape input: the label most clearly inside at its best sample
            // -- the centroid and a point toward each face. A knife edge's wedge tet
            // (a box less a larger sphere: at its holes' rims) has its centroid a
            // hair inside the sphere, whose side is chords; on its flat side it is
            // plainly solid. A tet in the hole is outside at every sample.
            double sp[5][3];

            for (int k = 0; k < 3; ++k) {
                sp[0][k] = o[k];
            }

            const double* vv[4] = { a, b, c, e };

            for (int f = 0; f < 4; ++f)
                for (int k = 0; k < 3; ++k) {
                    double fc = 0;

                    for (int j = 0; j < 4; ++j)
                        if (j != f) {
                            fc += vv[j][k] / 3.0;
                        }

                    sp[f + 1][k] = fc + 0.25 * (o[k] - fc);
                }

            int best = I[0];
            float bm = -1e30f;

            for (int x = 0; x < ni; ++x) {
                float mx = -1e30f;

                for (int q = 0; q < 5; ++q) {
                    const float px = static_cast<float>(sp[q][0]), py = static_cast<float>(sp[q][1]), pz = static_cast<float>(sp[q][2]);
                    const float wx = v2m_phi_f(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(),
                                               g.gI, g.gTW.data(), g.gm, I[x], px, py, pz);
                    float wo = -1e30f;

                    for (int y = 0; y < ni; ++y)
                        if (y != x) {
                            wo = std::max(wo, v2m_phi_f(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                                        g.phi.data(), g.gI, g.gTW.data(), g.gm, I[y], px, py, pz));
                        }

                    mx = std::max(mx, wx - wo);
                }

                if (mx > bm) {
                    bm = mx;
                    best = I[x];
                }
            }

            tl[t] = best;
        } else {   // several (a flat tet on one interface / junction): the centroid decides
            int best = I[0];
            float pb = -1.0f;

            for (int x = 0; x < ni; ++x) {
                if (I[x] == lc) {
                    best = lc;
                    break;
                }

                const float w = v2m_phi_f(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                         g.phi.data(), g.gI, g.gTW.data(), g.gm, I[x], static_cast<float>(o[0]),
                                         static_cast<float>(o[1]), static_cast<float>(o[2]));

                if (w > pb) {
                    pb = w;
                    best = I[x];
                }
            }

            tl[t] = best;
        }
    }

    st.ms_label = since(t1);
    lap("label");

    // 2b. sculpting. Delaunay fills the convex hull: across a concavity, and as
    // flat "kite" tets under a convex surface, it builds tets whose four nodes all
    // lie on the exterior surface. Removing such a tet can never expose an interior
    // node, so boundary tets (next to a removed tet or the hull) whose 4 nodes are
    // all exterior-surface nodes are peeled, round by round, while they are flat
    // (Joe-Liu < 0.2), have an edge through label 0, or have their circumcentre
    // outside.
    auto on_exterior = [&](uint32_t v) {
        int S[4];
        const int n = node_label_set(nd, v, S);

        for (int x = 1; x < n; ++x)
            if (S[x] == 0) {
                return true;
            }

        return false;
    };
    const float vmin0 = std::min(g.vs[0], std::min(g.vs[1], g.vs[2]));
    // Depth (mm) of a label-0 sample beyond the surface of label `sec`: psi / |grad
    // psi| with psi = phi_0 - phi_sec. A faceted mesh of a CONCAVE surface always
    // has chords a sagitta h^2/8R outside, so only depth > 0.2 h counts.
    // Is a sample deeper than `thr` (mm) below the exterior surface? From the
    // VOXELS (exact and independent of the interface field; |psi| / |grad psi|
    // blows up in the sharp thin-layer field): never in a non-exterior voxel, else
    // by the distance to the nearest face between an exterior voxel and a TISSUE
    // voxel -- of `sec` (the runner-up) first, then of any other label: near a
    // junction of the exterior with two tissues the nearest face need not be the
    // runner-up's (ANTS --tpm-fields: 16 short chords between 0|4|5 junction
    // nodes, practically on the surface, were "deep" by the 0|5 faces alone and
    // blocked the repairs every round)
    auto deep_outside = [&](float x, float y, float z, int sec, float thr) {
        const int vi = static_cast<int>(std::floor(x / g.vs[0] + 0.5f)), vj = static_cast<int>(std::floor(y / g.vs[1] + 0.5f)),
                  vk = static_cast<int>(std::floor(z / g.vs[2] + 0.5f));

        if (v2m_label_at(g.L->data(), g.nx, g.ny, g.nz, vi, vj, vk) != 0) {
            return false;
        }

        if (sec == V2M_NOLAB || sec == 0) {
            return true;
        }

        const float q[3] = { x, y, z };
        float yv[3];

        for (int l = 0; l < g.nlab; ++l) {
            const int b = l == 0 ? sec : l;   // the runner-up first

            if (l > 0 && (b == sec || b == 0)) {
                continue;
            }

            const float d2 = v2m_vox_nearest(d, g.L->data(), 0, b, V2M_NOLAB, q, 3, yv);

            if (d2 >= 0.0f && d2 <= thr * thr) {
                return false;
            }
        }

        return true;
    };
    auto v2m_h_at_voxel = [&](const float* p) {
        int i = static_cast<int>(std::floor(p[0] / g.vs[0] + 0.5f)), j = static_cast<int>(std::floor(p[1] / g.vs[1] + 0.5f)),
            k = static_cast<int>(std::floor(p[2] / g.vs[2] + 0.5f));
        i = std::min(std::max(i, 0), g.nx - 1);
        j = std::min(std::max(j, 0), g.ny - 1);
        k = std::min(std::max(k, 0), g.nz - 1);
        return g.h[i + static_cast<size_t>(g.nx) * (j + static_cast<size_t>(g.ny) * k)];
    };
    auto edge_outside = [&](uint32_t x, uint32_t y) {
        const float* p = &nd.P[3 * x];
        const float* q = &nd.P[3 * y];
        const float len = std::sqrt((q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]));
        const int ns = std::max(2, static_cast<int>(std::ceil(4.0f * len / vmin0)));

        for (int s = 1; s < ns; ++s) {
            const float tt = static_cast<float>(s) / ns;
            const float sx = p[0] + tt * (q[0] - p[0]), sy = p[1] + tt * (q[1] - p[1]), sz = p[2] + tt * (q[2] - p[2]);
            {   // cheap pre-check: label 0 must be in the sample's brick label list
                const int vi = std::min(g.nx - 1, std::max(0, static_cast<int>(std::floor(sx / g.vs[0] + 0.5f)))),
                          vj = std::min(g.ny - 1, std::max(0, static_cast<int>(std::floor(sy / g.vs[1] + 0.5f)))),
                          vk = std::min(g.nz - 1, std::max(0, static_cast<int>(std::floor(sz / g.vs[2] + 0.5f))));
                const int b = v2m_brick_of(d, vi, vj, vk);
                bool has0 = false;

                for (int e = 0; e < g.bl_cnt[b] && e < V2M_BL; ++e) {
                    has0 |= g.bl_lab[static_cast<size_t>(b) * V2M_BL + e] == 0;
                }

                if (!has0) {
                    continue;
                }
            }
            int sec;
            float mg;
            const int l = v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm,
                                      voxel_mode ? 1 : 0, p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]),
                                      p[2] + tt * (q[2] - p[2]), &sec, &mg);

            if (l == 0 && deep_outside(p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]),
                                       sec, 0.2f * v2m_h_at_voxel(p) + 0.87f * vmin0)) {   // + the staircase half-diagonal
                return true;
            }
        }

        return false;
    };
    auto peelable = [&](int64_t t) {
        const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
        bool all_ext = true;

        for (int k = 0; k < 4; ++k) {
            all_ext = all_ext && on_exterior(v[k]);
        }

        if (all_ext && surface_only) {
            // only the surface nodes (--mode surface): every node is on a surface,
            // so a peeled tet exposes the next, and flatness or the circumcentre
            // (all but arbitrary for nodes on one sphere) would eat into the
            // region layer by layer -- a hollow shell whose inner side pinches
            // the surface. Only a tet that lies outside is peeled: its centroid
            // exterior, or an edge through label 0 (the tets are not kept:
            // their shape does not matter)
            float c[3] = { 0, 0, 0 };

            for (int k = 0; k < 4; ++k)
                for (int e = 0; e < 3; ++e) {
                    c[e] += 0.25f * static_cast<float>(X[3 * v[k] + e]);
                }

            int sec;
            float mg;

            if (v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI,
                            g.gTW.data(), g.gm, voxel_mode ? 1 : 0, c[0], c[1], c[2], &sec, &mg) == 0) {
                return true;
            }

            for (int a = 0; a < 4; ++a)
                for (int b = a + 1; b < 4; ++b)
                    if (edge_outside(v[a], v[b])) {
                        return true;
                    }

            return false;
        }

        if (!all_ext) {
            // a tet bridging an exterior channel to a node that is not on the
            // exterior surface (a thin skin over another layer): peeled only if its
            // centroid lies deep outside (> 0.5 h past the voxel surface); the face
            // it exposes is then refined by the face repairs
            float c[3] = { 0, 0, 0 };

            for (int k = 0; k < 4; ++k)
                for (int e = 0; e < 3; ++e) {
                    c[e] += 0.25f * static_cast<float>(X[3 * v[k] + e]);
                }

            int sec;
            float mg;

            if (v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI,
                            g.gTW.data(), g.gm, voxel_mode ? 1 : 0, c[0], c[1], c[2], &sec, &mg) != 0) {
                return false;
            }

            return deep_outside(c[0], c[1], c[2], sec, 0.5f * v2m_h_at_voxel(c) + 0.87f * vmin0);
        }

        double q[4][3];
        const double* pp[4];

        for (int k = 0; k < 4; ++k) {
            for (int c = 0; c < 3; ++c) {
                q[k][c] = X[3 * v[k] + c];
            }

            pp[k] = q[k];
        }

        double md, jl, vol;
        tet_quality(pp, md, jl, vol);

        if (!g.feat.empty() && jl > 0.05) {
            // shape input: a tet holding two pinned feature nodes carries a protected
            // crease edge (a knife edge's rim: its tets are thin, their chords dip
            // outside) -- kept, unless it is flat (a kite on a face: nothing)
            int pins = 0;

            for (int k = 0; k < 4; ++k) {
                pins += v[k] < nd.size() && nd.typ[v[k]] == V2M_CORNER;
            }

            if (pins >= 2) {
                return false;
            }
        }

        if (jl < 0.2) {
            return true;
        }

        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b)
                if (edge_outside(v[a], v[b])) {
                    return true;
                }

        double o[3];

        if (circumcentre(pp[0], pp[1], pp[2], pp[3], o)) {
            int sec;
            float mg;

            if (v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm,
                            voxel_mode ? 1 : 0, static_cast<float>(o[0]), static_cast<float>(o[1]),
                            static_cast<float>(o[2]), &sec, &mg) == 0) {
                return true;
            }
        }

        return false;
    };
    st.peeled = 0;

    // (v2mesh: round 0 scans every tet in parallel; afterwards only the neighbours
    // of the tets just peeled can have become boundary tets -- peelable() depends
    // on the tet alone, so a boundary tet kept once stays kept: the same result)
    std::vector<int64_t> peeled_last;
    auto is_bnd = [&](int64_t t) {
        const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);

        for (int f = 0; f < 4; ++f)
            if (tl[nb[f] >> 2] <= 0) {
                return true;
            }

        return false;
    };

    for (int round = 0; round < 64; ++round) {
        std::vector<int64_t> cand;

        if (round == 0) {
            std::vector<std::vector<int64_t>> part(static_cast<size_t>(omp_get_max_threads()));
            #pragma omp parallel
            {
                std::vector<int64_t>& mine = part[static_cast<size_t>(omp_get_thread_num())];
                #pragma omp for schedule(static)

                for (int64_t t = 0; t < nt; ++t)
                    if (tl[t] > 0 && is_bnd(t)) {
                        mine.push_back(t);
                    }
            }

            for (auto& q : part) {
                cand.insert(cand.end(), q.begin(), q.end());
            }
        } else {
            for (int64_t t : peeled_last) {
                const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);

                for (int f = 0; f < 4; ++f) {
                    const int64_t u = static_cast<int64_t>(nb[f] >> 2);

                    if (tl[u] > 0) {
                        cand.push_back(u);
                    }
                }
            }

            std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        }

        std::vector<char> rm(cand.size(), 0);
        #pragma omp parallel for schedule(monotonic: dynamic, 256)

        for (int64_t c = 0; c < static_cast<int64_t>(cand.size()); ++c) {
            rm[c] = peelable(cand[c]);
        }

        size_t np = 0;
        peeled_last.clear();

        for (size_t c = 0; c < cand.size(); ++c)
            if (rm[c]) {
                tl[cand[c]] = 0;
                peeled_last.push_back(cand[c]);
                ++np;
            }

        st.peeled += np;

        if (np == 0) {
            break;
        }
    }

    lap("sculpt");

    // 2c. surfaces only: a flat tet whose four nodes all lie on one interface
    // takes its label from the smoothed field at its centroid -- on the interface
    // itself, so either side, by a hair. The wrong one leaves a fin standing on
    // the surface or a pocket in it, pinching the surface along an edge (edges
    // on 4 faces). Such a tet takes the label 3 of its 4 neighbours share, if
    // every node of it has that label too (the surface stays on the nodes)
    if (surface_only) {
        for (int round = 0; round < 8; ++round) {
            std::vector<int> nl(tl);
            size_t changed = 0;
            #pragma omp parallel for schedule(static) reduction(+ : changed)

            for (int64_t t = 0; t < nt; ++t) {
                if (tl[t] < 0) {
                    continue;   // a ghost
                }

                const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);
                int lb[4];

                for (int f = 0; f < 4; ++f) {
                    lb[f] = std::max(0, tl[nb[f] >> 2]);   // (the hull: the exterior)
                }

                for (int f = 0; f < 4; ++f) {
                    int cnt = 0;

                    for (int e = 0; e < 4; ++e) {
                        cnt += lb[e] == lb[f];
                    }

                    if (cnt < 3 || lb[f] == tl[t]) {
                        continue;
                    }

                    const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
                    bool all = true;

                    for (int k = 0; k < 4 && all; ++k) {
                        int S[4];
                        const int n = node_label_set(nd, v[k], S);
                        bool has = false;

                        for (int x = 0; x < n; ++x) {
                            has = has || S[x] == lb[f];
                        }

                        all = has;
                    }

                    if (all) {
                        nl[t] = lb[f];
                        ++changed;
                    }

                    break;
                }
            }

            tl.swap(nl);

            if (changed == 0) {
                break;
            }
        }
    }

    // 3. kept tets and the conformity checks
    clk::time_point t2 = clk::now();
    m.P = nd.P;
    {   // (v2mesh: parallel compaction, same order)
        std::vector<int64_t> pos(static_cast<size_t>(nt) + 1, 0);
        size_t ndt = 0;
        #pragma omp parallel for schedule(static) reduction(+ : ndt)

        for (int64_t t = 0; t < nt; ++t) {
            pos[t + 1] = tl[t] > 0 ? 1 : 0;
            ndt += tl[t] >= 0;
        }

        for (int64_t t = 0; t < nt; ++t) {
            pos[t + 1] += pos[t];
        }

        st.delaunay_tets = ndt;
        m.tets.resize(static_cast<size_t>(pos[nt]) * 4);
        m.label.resize(static_cast<size_t>(pos[nt]));
        #pragma omp parallel for schedule(static)

        for (int64_t t = 0; t < nt; ++t) {
            if (tl[t] <= 0) {
                continue;
            }

            const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
            const int64_t w = pos[t];

            for (int k = 0; k < 4; ++k) {
                m.tets[4 * w + k] = static_cast<int32_t>(v[k]);
            }

            m.label[w] = tl[t];
        }
    }

    st.kept = m.label.size();
    // node is on the interface between labels x and y?
    auto on_iface = [&](int i, int x, int y) {
        if (nd.typ[i] == V2M_INTERIOR) {
            return false;
        }

        int S[4];
        const int n = node_label_set(nd, static_cast<uint32_t>(i), S);
        bool hx = false, hy = false;

        for (int k = 0; k < n; ++k) {
            hx |= S[k] == x;
            hy |= S[k] == y;
        }

        return hx && hy;
    };
    size_t bad_faces = 0, bad_span = 0;
    std::vector<Fix> ffix;   // bad faces: put an a|b node at the face centroid
    FILE* dbg = std::getenv("V2M_TESS_DEBUG") ? std::fopen(std::getenv("V2M_TESS_DEBUG"), "wb") : nullptr;

    if (!dbg) {   // (v2mesh: parallel face check; per-thread lists concatenated in tet order)
        std::vector<std::vector<Fix>> part(static_cast<size_t>(omp_get_max_threads()));
        size_t bf = 0, bs = 0;
        #pragma omp parallel reduction(+ : bf, bs)
        {
            std::vector<Fix>& mine = part[static_cast<size_t>(omp_get_thread_num())];
            #pragma omp for schedule(static)

            for (int64_t t = 0; t < nt; ++t) {
                if (tl[t] <= 0) {
                    continue;
                }

                const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
                bs += conflict[t];
                const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);

                for (int f = 0; f < 4; ++f) {
                    const int64_t u = static_cast<int64_t>(nb[f] >> 2);
                    const int lu = tl[u] < 0 ? 0 : tl[u];

                    if (lu == tl[t] || (lu != 0 && u < t)) {
                        continue;
                    }

                    for (int k = 0; k < 4; ++k)
                        if (k != f && !on_iface(static_cast<int>(v[k]), tl[t], lu)) {
                            ++bf;
                            Fix fx;
                            fx.x = static_cast<uint32_t>(t);
                            fx.y = V2M_FACEFIX;
                            fx.a = tl[t];
                            fx.b = lu;

                            for (int e = 0, m2 = 0; e < 4; ++e)
                                if (e != f) {
                                    fx.tv[m2++] = v[e];
                                }

                            fx.tv[3] = UINT32_MAX;
                            mine.push_back(fx);
                            break;
                        }
                }
            }
        }

        for (auto& q : part) {
            ffix.insert(ffix.end(), q.begin(), q.end());
        }

        bad_faces = bf;
        bad_span = bs;
    }

    for (int64_t t = 0; dbg && t < nt; ++t) {
        if (tl[t] <= 0) {
            continue;
        }

        const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);

        bad_span += conflict[t];

        if (dbg) {   // debug: x y z (centroid) kind label; kind 1 = spanning, 2 = bad face
            float r[7] = { 0, 0, 0, 0, static_cast<float>(tl[t]), 0, 0 };
            bool bf = false;
            const uint64_t* nb2 = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);

            for (int f = 0; f < 4 && !bf; ++f) {
                const int64_t u = static_cast<int64_t>(nb2[f] >> 2);
                const int lu = tl[u] < 0 ? 0 : tl[u];

                if (lu != tl[t]) {
                    for (int k = 0; k < 4; ++k)
                        if (k != f && !on_iface(static_cast<int>(v[k]), tl[t], lu)) {
                            bf = true;
                        }
                }
            }

            if (conflict[t] || bf) {
                for (int k = 0; k < 4; ++k)
                    for (int e = 0; e < 3; ++e) {
                        r[e] += 0.25f * static_cast<float>(X[3 * v[k] + e]);
                    }

                r[3] = conflict[t] ? 1.0f : 2.0f;
                int U[12], nu = 0, ninter = 0;   // union of the node label sets; interior nodes

                for (int k = 0; k < 4; ++k) {
                    int S2[4];
                    const int n2 = node_labels(v[k], S2);
                    ninter += nd.typ[v[k]] == V2M_INTERIOR;

                    for (int e = 0; e < n2; ++e) {
                        bool seen = false;

                        for (int x = 0; x < nu; ++x) {
                            seen |= U[x] == S2[e];
                        }

                        if (!seen) {
                            U[nu++] = S2[e];
                        }
                    }
                }

                r[5] = static_cast<float>(nu);
                r[6] = static_cast<float>(ninter);
                std::fwrite(r, sizeof(float), 7, dbg);

                if (std::getenv("V2M_TESS_VERBOSE")) {   // the nodes of every failing tet
                    int sc;
                    float mg;
                    const int lc2 = v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                                g.phi.data(), g.gI, g.gTW.data(), g.gm, 0, r[0], r[1], r[2], &sc, &mg);
                    std::fprintf(stderr, "[fail] tet %lld %s label %d, centroid (%.2f %.2f %.2f) field %d\n",
                                 static_cast<long long>(t), conflict[t] ? "SPAN" : "FACE", tl[t], r[0], r[1], r[2], lc2);

                    for (int k = 0; k < 4; ++k) {
                        int S3[4];
                        const int n3 = node_labels(v[k], S3);
                        const float* q = &nd.P[3 * v[k]];
                        const int lq = v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                                   g.phi.data(), g.gI, g.gTW.data(), g.gm, 0, q[0], q[1], q[2], &sc, &mg);
                        std::fprintf(stderr, "        node %u typ %d set {", v[k], nd.typ[v[k]]);

                        for (int e = 0; e < n3; ++e) {
                            std::fprintf(stderr, "%s%d", e ? "," : "", S3[e]);
                        }

                        std::fprintf(stderr, "} at (%.2f %.2f %.2f) field %d (2nd %d, margin %.3f)\n", q[0], q[1], q[2], lq, sc,
                                     mg);
                    }
                }
            }
        }

        const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);

        for (int f = 0; f < 4; ++f) {
            const int64_t u = static_cast<int64_t>(nb[f] >> 2);
            const int lu = tl[u] < 0 ? 0 : tl[u];

            if (lu == tl[t] || (lu != 0 && u < t)) {
                continue;    // same label, or an internal face counted from the lower tet
            }

            for (int k = 0; k < 4; ++k)
                if (k != f && !on_iface(static_cast<int>(v[k]), tl[t], lu)) {
                    ++bad_faces;
                    Fix fx;
                    fx.x = static_cast<uint32_t>(t);
                    fx.y = V2M_FACEFIX;
                    fx.a = tl[t];
                    fx.b = lu;

                    for (int e = 0, m2 = 0; e < 4; ++e)
                        if (e != f) {
                            fx.tv[m2++] = v[e];
                        }

                    fx.tv[3] = UINT32_MAX;
                    ffix.push_back(fx);
                    break;
                }
        }
    }

    if (dbg) {
        std::fclose(dbg);
    }

    lap("faces");

    // (b) kept edges whose segment passes through label 0 (sampled every 1/4 voxel)
    const std::vector<std::pair<int, int>> edges = unique_edges(m.tets, n);
    lap("uniq-edges");
    std::vector<char> eout(edges.size(), 0);   // cached for the repairs below
    // an insert-only round (first_new >= 0) changes only the stars of the new
    // nodes: an edge between two nodes outside those stars keeps last round's
    // result (positions are unchanged), so only the "dirty" edges are re-tested
    std::vector<char> dirty;

    if (first_new >= 0) {
        dirty.assign(n, 0);
        #pragma omp parallel for schedule(static)

        for (int64_t t = 0; t < static_cast<int64_t>(m.label.size()); ++t) {
            bool hit = false;

            for (int k = 0; k < 4; ++k) {
                hit |= m.tets[4 * t + k] >= first_new;
            }

            if (hit) {
                for (int k = 0; k < 4; ++k) {
                    dirty[m.tets[4 * t + k]] = 1;
                }
            }
        }
    }

    lap("dirty");
    size_t bad_edges = 0;
    #pragma omp parallel for schedule(monotonic: dynamic, 4096) reduction(+ : bad_edges)

    for (int64_t e = 0; e < static_cast<int64_t>(edges.size()); ++e) {
        const int x = edges[e].first, y = edges[e].second;

        if (first_new >= 0 && !dirty[x] && !dirty[y]) {
            eout[e] = std::binary_search(eout_prev.begin(), eout_prev.end(), edges[e]) ? 1 : 0;
        } else {
            eout[e] = edge_outside(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) ? 1 : 0;
        }

        bad_edges += eout[e];
    }

    eout_prev.clear();

    for (size_t e = 0; e < edges.size(); ++e)
        if (eout[e]) {
            eout_prev.push_back(edges[e]);   // sorted: unique_edges emits (x, y) ascending
        }

    lap("edges-out");

    // repairs: every edge of a crossing tet whose endpoints share no label, and
    // every kept edge through label 0
    fixes.clear();
    std::vector<Fix> jfix;
    {
        std::vector<std::pair<uint32_t, uint32_t>> seen;

        for (int64_t t = 0; t < nt; ++t) {
            if (tl[t] <= 0 || !conflict[t]) {
                continue;
            }

            const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
            bool any_disjoint = false;

            for (int a = 0; a < 4; ++a)
                for (int b = a + 1; b < 4; ++b) {
                    int Sa[4], Sb[4];
                    const int na = node_labels(v[a], Sa), nb2 = node_labels(v[b], Sb);
                    bool share = false;

                    for (int x = 0; x < na && !share; ++x)
                        for (int y = 0; y < nb2; ++y)
                            if (Sa[x] == Sb[y]) {
                                share = true;
                                break;
                            }

                    if (!share) {
                        const uint32_t x = std::min(v[a], v[b]), y = std::max(v[a], v[b]);
                        seen.emplace_back(x, y);
                        any_disjoint = true;
                    }
                }

            // every pair shares a label but no label is common to all four: the tet
            // straddles a junction curve with no junction node near -> add one
            if (!any_disjoint) {
                Fix f;
                f.x = f.y = UINT32_MAX;
                f.a = f.b = -1;

                for (int k = 0; k < 4; ++k) {
                    f.tv[k] = v[k];
                }

                jfix.push_back(f);
            }
        }

        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());

        for (const std::pair<uint32_t, uint32_t>& e : seen) {
            Fix f;
            f.x = e.first;
            f.y = e.second;
            f.a = nd.lab[e.first];
            f.b = nd.lab[e.second];
            fixes.push_back(f);
        }

        fixes.insert(fixes.end(), jfix.begin(), jfix.end());
        // face fixes are kept apart: tessellate() uses them only once the crossing /
        // junction repairs are exhausted (most bad faces vanish with those, and
        // adding both over-refines: wedge +3k nodes)
        facefixes.swap(ffix);

        for (int64_t e = 0; e < static_cast<int64_t>(edges.size()); ++e)
            if (eout[e]) {
                Fix f;
                f.x = static_cast<uint32_t>(edges[e].first);
                f.y = static_cast<uint32_t>(edges[e].second);
                f.a = nd.lab[edges[e].first];
                f.b = 0;
                fixes.push_back(f);
            }
    }

    lap("gather");
    st.bad_faces = bad_faces;
    st.bad_edges = bad_edges;
    st.bad_span = bad_span;

    // the spanning tets of this round, for the deviation metric (measured once,
    // after the repairs, by deviation_metrics)
    span_tets.clear();

    for (int64_t t = 0; t < nt; ++t)
        if (tl[t] > 0 && conflict[t]) {
            const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
            span_tets.push_back({ { v[0], v[1], v[2], v[3], static_cast<uint32_t>(tl[t]) } });
        }

    st.ms_check = since(t2);
    lap("deviation");

}


// Apply the repairs: the crossing point of each edge with its interface (the
// exterior surface for b = 0) is projected onto it; if an endpoint is within
// 0.3 h it is PROMOTED onto the interface (moved there), else a new interface
// (or junction) node is added. New points closer than 0.3 h to one already added
// this round are skipped.
// every node in a hash grid of cell 0.3 hmin, kept across the repair rounds: the
// node set only grows between them, so each call appends the new nodes (a full
// rebuild of the ~1M-node map cost ~150 ms per call); invalidated when nodes move
// or are dropped (a promotion, a rolled-back -q round). Cells list their nodes in
// index order either way, so the lookups are the same as a fresh build's.
struct NodeGrid {
    float cell = 0.0f;
    size_t n = 0;
    bool valid = false;
    std::unordered_map<int64_t, std::vector<uint32_t>> map;
};

// allow_move = false: insertions only (no junction promotion; `promote` still
// moves endpoints): the round keeps the live Delaunay incremental
static size_t apply_fixes(const Grid& g, const std::vector<Fix>& fixes, Nodes& nd, size_t* moved, bool promote,
                          NodeGrid& G, bool allow_move = true) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
#define FLD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm
    std::vector<char> touched(nd.size(), 0);
    size_t nfix = 0, sk_same = 0, sk_out = 0, sk_sign = 0, sk_near = 0;
    // a new or promoted position must keep >= 0.3 h from all other nodes
    // (coincident points break the exact Delaunay's symbolic perturbation)
    const float cell = 0.3f * g.hmin;
    std::unordered_map<int64_t, std::vector<uint32_t>>& grid = G.map;

    if (!G.valid || G.cell != cell || G.n > nd.size()) {
        grid.clear();
        G.cell = cell;
        G.n = 0;
        G.valid = true;
    }

    auto ckey = [&](const float* x) {
        const int64_t i = static_cast<int64_t>(std::floor(x[0] / cell)), j = static_cast<int64_t>(std::floor(x[1] / cell)),
                      k = static_cast<int64_t>(std::floor(x[2] / cell));
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };

    for (uint32_t v = static_cast<uint32_t>(G.n); v < nd.size(); ++v) {
        grid[ckey(&nd.P[3 * v])].push_back(v);
    }

    // the moved nodes and the key of the cell they left (their grid entries are
    // fixed at the end: the old one dropped, the new cell back in index order)
    std::vector<std::pair<uint32_t, int64_t>> moved_from;

    // nearest other node within r of x (skip `self`), or -1
    auto near_node = [&](const float* x, float r, int64_t self0, int64_t self1) -> int64_t {
        const int m = static_cast<int>(std::ceil(r / cell));

        for (int dz = -m; dz <= m; ++dz)
            for (int dy = -m; dy <= m; ++dy)
                for (int dx = -m; dx <= m; ++dx) {
                    const float y[3] = { x[0] + dx * cell, x[1] + dy * cell, x[2] + dz * cell };
                    std::unordered_map<int64_t, std::vector<uint32_t>>::const_iterator it = grid.find(ckey(y));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (uint32_t v : it->second) {
                        if (v == self0 || v == self1) {
                            continue;
                        }

                        const float ex = nd.P[3 * v] - x[0], ey = nd.P[3 * v + 1] - x[1], ez = nd.P[3 * v + 2] - x[2];

                        if (ex * ex + ey * ey + ez * ez < r * r) {
                            return v;
                        }
                    }
                }

        return -1;
    };

    size_t njf = 0, nff = 0, sk_jf = 0, sk_ff = 0;
    // a new node must be strictly valid: the field's argmax at its position is one
    // of its labels (the checks and the repair walks use the strict argmax; a node
    // off its labels blocks the repair of every edge it gets)
    // (probability fields only: with sharp indicator fields the relaxation's 0.02
    // tolerance is harmless, and the strict test only rejected useful repairs)
    auto strict_ok = [&](const float* x, int a, int b, int c) {
        if (!g.prob_fields) {
            return true;
        }

        int sec;
        float mg;
        const int l = v2m_label_of(FLD, 0, x[0], x[1], x[2], &sec, &mg);
        return l == a || l == b || (c != V2M_NOLAB && l == c);
    };

    // A repair point on a|b refused for a node within 0.3 h: if that node is an
    // interface node of another pair making a junction with a|b ({a, b} and its
    // own labels: 3 in all), it is promoted onto the junction curve instead --
    // near a junction the two interfaces' nodes crowd, and neither side's repair
    // could otherwise land (a moved node: the next round rebuilds the Delaunay)
    auto promote_junction = [&](int v, int a, int b, float h) {
        if (!allow_move || v < 0 || touched[static_cast<size_t>(v)] || nd.typ[static_cast<size_t>(v)] != V2M_INTERFACE) {
            return false;
        }

        int U[3] = { a, b, V2M_NOLAB }, nu = 2;
        const int own[2] = { nd.lab[static_cast<size_t>(v)], nd.part[2 * static_cast<size_t>(v)] };

        for (int x : own) {
            if (x == U[0] || x == U[1] || (nu > 2 && x == U[2])) {
                continue;
            }

            if (nu == 3) {
                return false;   // (4 labels: not a junction)
            }

            U[nu++] = x;
        }

        if (nu != 3) {
            return false;
        }

        // the own label: the node's, never 0
        int ja = own[0] != 0 ? own[0] : own[1], jb = -1, jc = -1;

        for (int x : U)
            if (x != ja) {
                (jb < 0 ? jb : jc) = x;
            }

        float r[3] = { nd.P[3 * static_cast<size_t>(v)], nd.P[3 * static_cast<size_t>(v) + 1], nd.P[3 * static_cast<size_t>(v) + 2] };

        // (a junction point, where three interfaces' nodes crowd: 0.15 h spacing)
        if (!v2m_project2(FLD, ja, jb, jc, r, 0.5f * h) || !v2m_valid_on(FLD, ja, jb, jc, r) || !strict_ok(r, ja, jb, jc) ||
                near_node(r, 0.15f * h, v, v) >= 0) {
            if (std::getenv("V2M_REPAIR_DEBUG")) {
                float r2[3] = { nd.P[3 * static_cast<size_t>(v)], nd.P[3 * static_cast<size_t>(v) + 1], nd.P[3 * static_cast<size_t>(v) + 2] };
                const int p2 = v2m_project2(FLD, ja, jb, jc, r2, 0.5f * h);
                std::fprintf(stderr, "[promote] node %d {%d,%d,%d}: project2 %d valid %d strict %d near %lld (moved %.2f)\n", v, ja, jb,
                             jc, p2, p2 ? v2m_valid_on(FLD, ja, jb, jc, r2) : -1, p2 ? strict_ok(r2, ja, jb, jc) : -1,
                             p2 ? static_cast<long long>(near_node(r2, 0.15f * h, v, v)) : -2LL,
                             std::sqrt((r2[0] - nd.P[3 * v]) * (r2[0] - nd.P[3 * v]) + (r2[1] - nd.P[3 * v + 1]) * (r2[1] - nd.P[3 * v + 1]) +
                                       (r2[2] - nd.P[3 * v + 2]) * (r2[2] - nd.P[3 * v + 2])));
            }

            return false;
        }

        moved_from.push_back({ static_cast<uint32_t>(v), ckey(&nd.P[3 * static_cast<size_t>(v)]) });
        nd.P[3 * static_cast<size_t>(v)] = r[0];
        nd.P[3 * static_cast<size_t>(v) + 1] = r[1];
        nd.P[3 * static_cast<size_t>(v) + 2] = r[2];
        nd.lab[static_cast<size_t>(v)] = static_cast<uint16_t>(ja);
        nd.typ[static_cast<size_t>(v)] = V2M_JUNCTION;
        nd.part[2 * static_cast<size_t>(v)] = static_cast<uint16_t>(jb);
        nd.part[2 * static_cast<size_t>(v) + 1] = static_cast<uint16_t>(jc);
        touched[static_cast<size_t>(v)] = 1;
        grid[ckey(r)].push_back(static_cast<uint32_t>(v));
        ++*moved;
        return true;
    };

    for (const Fix& f : fixes) {
        if (f.y == V2M_FACEFIX) {   // the face centroid, projected onto the a|b interface
            const int a = f.a == 0 ? f.b : f.a, b = f.a == 0 ? 0 : f.b;
            float c[3] = { 0, 0, 0 };

            for (int k = 0; k < 3; ++k)
                for (int e = 0; e < 3; ++e) {
                    c[e] += nd.P[3 * f.tv[k] + e] / 3.0f;
                }

            const float h = v2m_h_at(d, g.h.data(), c[0], c[1], c[2]);

            if (a == b || !v2m_project1(FLD, a, b, c, 0.5f * h) || !v2m_valid_on(FLD, a, b, V2M_NOLAB, c) ||
                    !strict_ok(c, a, b, V2M_NOLAB)) {
                ++sk_ff;
                continue;
            }

            {
                const int nv = static_cast<int>(near_node(c, 0.3f * h, -1, -1));

                if (nv >= 0) {
                    if (promote_junction(nv, a, b, h)) {
                        ++njf;
                        ++nfix;
                    } else {
                        ++sk_ff;
                    }

                    continue;
                }
            }

            int typ = V2M_INTERFACE, cc = V2M_NOLAB;
            const int third = v2m_third_label(FLD, a, b, c);

            if (third != V2M_NOLAB) {
                float r[3] = { c[0], c[1], c[2] };

                if (v2m_project2(FLD, a, b, third, r, 0.5f * h) && v2m_valid_on(FLD, a, b, third, r) &&
                        strict_ok(r, a, b, third) && near_node(r, 0.3f * h, -1, -1) < 0) {
                    c[0] = r[0];
                    c[1] = r[1];
                    c[2] = r[2];
                    typ = V2M_JUNCTION;
                    cc = third;
                }
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(a));
            nd.typ.push_back(static_cast<uint8_t>(typ));
            nd.part.push_back(static_cast<uint16_t>(b));
            nd.part.push_back(static_cast<uint16_t>(cc));
            nd.part3.push_back(V2M_NOLAB);
            touched.push_back(1);
            ++nff;
            ++nfix;
            continue;
        }

        if (f.y == UINT32_MAX) {   // junction fix: the 3 most frequent labels of the tet's nodes
            int cnt[24] = { 0 }, lab[24];
            int nl = 0;

            for (int k = 0; k < 4; ++k) {
                const uint32_t v = f.tv[k];
                int S[4];
                const int ns = node_label_set(nd, v, S);

                for (int x = 0; x < ns; ++x) {
                    int y = 0;

                    while (y < nl && lab[y] != S[x]) {
                        ++y;
                    }

                    if (y == nl) {
                        lab[nl] = S[x];
                        cnt[nl++] = 0;
                    }

                    ++cnt[y];
                }
            }

            if (nl < 3) {
                continue;
            }

            for (int x = 0; x < nl; ++x)      // sort by count, descending
                for (int y = x + 1; y < nl; ++y)
                    if (cnt[y] > cnt[x]) {
                        std::swap(cnt[x], cnt[y]);
                        std::swap(lab[x], lab[y]);
                    }

            int ja = lab[0], jb = lab[1], jc = lab[2];

            if (ja == 0) {   // the own label is never 0
                std::swap(ja, jc);
            }

            float c[3] = { 0, 0, 0 };

            for (int k = 0; k < 4; ++k)
                for (int e = 0; e < 3; ++e) {
                    c[e] += 0.25f * nd.P[3 * f.tv[k] + e];
                }

            const float h = v2m_h_at(d, g.h.data(), c[0], c[1], c[2]);

            if (!v2m_project2(FLD, ja, jb, jc, c, 0.5f * h) || !v2m_valid_on(FLD, ja, jb, jc, c) ||
                    !strict_ok(c, ja, jb, jc) || near_node(c, 0.3f * h, -1, -1) >= 0) {
                ++sk_jf;
                continue;
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(ja));
            nd.typ.push_back(V2M_JUNCTION);
            nd.part.push_back(static_cast<uint16_t>(jb));
            nd.part.push_back(static_cast<uint16_t>(jc));
            nd.part3.push_back(V2M_NOLAB);
            touched.push_back(1);
            ++njf;
            ++nfix;
            continue;
        }

        if (f.a == f.b && f.b != 0) {
            ++sk_same;
            continue;
        }

        float p[3], q[3], c[3];

        for (int k = 0; k < 3; ++k) {
            p[k] = nd.P[3 * f.x + k];
            q[k] = nd.P[3 * f.y + k];
        }

        if (f.b == 0) {
            // an edge through label 0 (node labels are never 0, so b = 0 marks these):
            // both endpoints may carry 0 in their sets, so the walk below would never
            // "leave" them. Refine the chord instead: its deepest exterior sample,
            // projected onto the surface of the runner-up tissue label there.
            const float len0 = std::sqrt((q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) +
                                         (q[2] - p[2]) * (q[2] - p[2]));
            const int ns0 = std::max(4, static_cast<int>(std::ceil(4.0f * len0 / std::min(g.vs[0], std::min(g.vs[1], g.vs[2])))));
            float best = -1.0f, c[3] = { 0, 0, 0 };
            int ls = V2M_NOLAB;

            for (int s = 1; s < ns0; ++s) {
                const float tt = static_cast<float>(s) / ns0;
                const float r[3] = { p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]) };
                int sec;
                float mg;

                if (v2m_label_of(FLD, 0, r[0], r[1], r[2], &sec, &mg) == 0 && sec != V2M_NOLAB && sec != 0 && mg > best) {
                    best = mg;   // margin of label 0 over the runner-up: the deepest sample
                    ls = sec;
                    c[0] = r[0];
                    c[1] = r[1];
                    c[2] = r[2];
                }
            }

            const float h0 = v2m_h_at(d, g.h.data(), c[0], c[1], c[2]);

            if (std::getenv("V2M_OUT_DEBUG") && sk_out < 5) {
                std::fprintf(stderr, "[outdbg] edge %u(typ %d lab %d part %d,%d at %.1f %.1f %.1f)-%u(typ %d lab %d part %d,%d) len %.2f best %.3f sec %d:",
                             f.x, nd.typ[f.x], nd.lab[f.x], nd.part[2 * f.x], nd.part[2 * f.x + 1], p[0], p[1], p[2], f.y,
                             nd.typ[f.y], nd.lab[f.y], nd.part[2 * f.y], nd.part[2 * f.y + 1], len0, best, ls);

                for (int s = 1; s < ns0; ++s) {
                    const float tt = static_cast<float>(s) / ns0;
                    const float r[3] = { p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]) };
                    int sec;
                    float mg;
                    const int l = v2m_label_of(FLD, 0, r[0], r[1], r[2], &sec, &mg);
                    std::fprintf(stderr, " %d/%d/%.2f", l, sec == V2M_NOLAB ? -1 : sec, mg);
                }

                std::fprintf(stderr, "\n");
            }

            // one placement: a start projected onto the l|0 surface; where a third
            // tissue wins there (a thin layer over it: the exterior surface is really
            // that tissue's), onto its surface instead. Kept if not near a node.
            // vox: when the projection from the start fails -- deep outside, the
            // smoothed fields are flat (both indicators saturated) and it has no
            // gradient to follow -- start instead from the nearest l|0 voxel face,
            // within a voxel of the smooth surface (a chord across an air gap between
            // two facing scalp sheets was left through label 0 in every round:
            // Colin27 --size 5 --relax fire)
            auto place = [&](const float* start, int l, float* out, int* lout, bool vox) {
                const float hs = v2m_h_at(d, g.h.data(), start[0], start[1], start[2]);
                float x[3] = { start[0], start[1], start[2] };

                if (!v2m_project1(FLD, l, 0, x, 0.5f * hs)) {
                    float y[3];

                    if (!vox) {
                        return false;
                    }

                    if (v2m_vox_nearest(d, g.L->data(), l, 0, V2M_NOLAB, start, 4, y) < 0.0f) {
                        return false;
                    }

                    x[0] = y[0];
                    x[1] = y[1];
                    x[2] = y[2];

                    if (!v2m_project1(FLD, l, 0, x, 0.5f * hs)) {
                        return false;
                    }
                }

                if (!strict_ok(x, l, 0, V2M_NOLAB)) {
                    int sec;
                    float mg;
                    const int l3 = v2m_label_of(FLD, 0, x[0], x[1], x[2], &sec, &mg);
                    x[0] = start[0];
                    x[1] = start[1];
                    x[2] = start[2];

                    if (l3 == 0 || l3 == l || !v2m_project1(FLD, l3, 0, x, 0.5f * hs) || !strict_ok(x, l3, 0, V2M_NOLAB)) {
                        return false;
                    }

                    l = l3;
                }

                if (near_node(x, 0.3f * hs, -1, -1) >= 0) {
                    return false;
                }

                out[0] = x[0];
                out[1] = x[1];
                out[2] = x[2];
                *lout = l;
                return true;
            };
            bool placed = false;

            // the deepest sample first, then the others, deepest first, if well
            // inside the exterior (margin >= 0.25: a shallow one only adds a node next
            // to the chord's ends) -- the deepest may not place: deep in an air gap
            // the fields are flat (p_0 in probability fields: face / jaw gaps on
            // ANTS), or its surface point already has a node that did not break the
            // chord. All by plain projection, then all from the voxel faces (which
            // first cost probability-field runs conformity: they placed the deepest
            // sample's node where a later sample's plain projection did better).
            if (best >= 0.0f) {
                std::vector<std::pair<float, int>> cand;

                for (int s = 1; s < ns0; ++s) {
                    const float tt = static_cast<float>(s) / ns0;
                    int sec;
                    float mg;

                    if (v2m_label_of(FLD, 0, p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]), &sec,
                                    &mg) == 0 && sec != V2M_NOLAB && sec != 0 && mg >= 0.25f) {
                        cand.push_back(std::make_pair(-mg, s));
                    }
                }

                std::sort(cand.begin(), cand.end());
                const float c0[3] = { c[0], c[1], c[2] };
                const int ls0 = ls;

                for (int vox = 0; vox < 2 && !placed; ++vox) {
                    placed = place(c0, ls0, c, &ls, vox != 0);

                    for (size_t k = 0; k < cand.size() && !placed; ++k) {
                        const float tt = static_cast<float>(cand[k].second) / ns0;
                        const float r[3] = { p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]) };
                        int sec;
                        float mg;
                        v2m_label_of(FLD, 0, r[0], r[1], r[2], &sec, &mg);
                        placed = place(r, sec, c, &ls, vox != 0);
                    }
                }
            }

            if (!placed) {
                ++sk_out;
                continue;
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(ls));
            nd.typ.push_back(V2M_INTERFACE);
            nd.part.push_back(0);
            nd.part.push_back(V2M_NOLAB);
            nd.part3.push_back(V2M_NOLAB);
            touched.push_back(1);
            ++nfix;
            continue;
        }

        // Walk the edge from x and bracket the first place where the label (argmax)
        // leaves x's label set: the crossing is on the interface between the last
        // label inside and the first outside -- which near a junction need not be
        // lab[x] | lab[y] (the edge can leave through a third label).
        int Sx[4];
        const int nsx = node_label_set(nd, f.x, Sx);

        const float len = std::sqrt((q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]));
        const int ns = std::max(4, static_cast<int>(std::ceil(4.0f * len / std::min(g.vs[0], std::min(g.vs[1], g.vs[2])))));
        float lo[3] = { p[0], p[1], p[2] }, hi[3];
        int la = -1, lb = -1;

        for (int s = 1; s <= ns; ++s) {
            const float tt = static_cast<float>(s) / ns;
            float r[3] = { p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]) };
            int sec;
            float mg;
            const int l = v2m_label_of(FLD, 0, r[0], r[1], r[2], &sec, &mg);
            bool in = false;

            for (int k = 0; k < nsx; ++k) {
                in = in || l == Sx[k];
            }

            if (!in) {
                lb = l;
                hi[0] = r[0];
                hi[1] = r[1];
                hi[2] = r[2];
                break;
            }

            la = l;
            lo[0] = r[0];
            lo[1] = r[1];
            lo[2] = r[2];
        }

        if (lb < 0) {
            ++sk_out;
            continue;
        }

        if (la < 0) {
            la = nd.lab[f.x];
        }

        const int a2 = la, b2 = lb;

        if (!(v2m_psi(FLD, a2, b2, lo[0], lo[1], lo[2]) > 0.0f && v2m_psi(FLD, a2, b2, hi[0], hi[1], hi[2]) <= 0.0f)) {
            ++sk_sign;
            continue;
        }

        for (int k = 0; k < 3; ++k) {
            p[k] = lo[k];
            q[k] = hi[k];
        }

        const int a = a2 == 0 ? b2 : a2, b = a2 == 0 ? 0 : b2;   // own label never 0
        v2m_crossing(FLD, a2, b2, p, q, c);
        const float h = v2m_h_at(d, g.h.data(), c[0], c[1], c[2]);
        v2m_project1(FLD, a, b, c, 0.25f * h);   // polish; the crossing is already on it
        int typ = V2M_INTERFACE, cc = V2M_NOLAB;
        const int third = v2m_third_label(FLD, a, b, c);

        if (third != V2M_NOLAB) {
            float r[3] = { c[0], c[1], c[2] };
            if (v2m_project2(FLD, a, b, third, r, 0.5f * h) && strict_ok(r, a, b, third)) {
                c[0] = r[0];
                c[1] = r[1];
                c[2] = r[2];
                typ = V2M_JUNCTION;
                cc = third;
            }
        }

        if (!strict_ok(c, a, b, cc)) {
            ++sk_sign;
            continue;
        }

        // promote an endpoint that is already close
        const uint32_t ends[2] = { f.x, f.y };
        bool done = false;

        // (first repair round only: a moved node forces a full Delaunay rebuild,
        // while insert-only rounds are incremental)

        for (int e = 0; e < 2 && !done && promote; ++e) {
            const uint32_t v = ends[e];
            const float dx = nd.P[3 * v] - c[0], dy = nd.P[3 * v + 1] - c[1], dz = nd.P[3 * v + 2] - c[2];

            if (!touched[v] && nd.typ[v] == V2M_INTERIOR && dx * dx + dy * dy + dz * dz < 0.09f * h * h &&
                    (nd.lab[v] == a || nd.lab[v] == b) && near_node(c, 0.3f * h, v, v) < 0) {
                ++*moved;
                moved_from.push_back({ v, ckey(&nd.P[3 * v]) });
                nd.P[3 * v] = c[0];
                nd.P[3 * v + 1] = c[1];
                nd.P[3 * v + 2] = c[2];
                nd.typ[v] = static_cast<uint8_t>(typ);
                nd.part[2 * v] = static_cast<uint16_t>(nd.lab[v] == a ? b : a);
                nd.part[2 * v + 1] = static_cast<uint16_t>(cc);
                touched[v] = 1;
                grid[ckey(c)].push_back(v);
                done = true;
            }
        }

        if (!done) {
            const int nv = static_cast<int>(near_node(c, 0.3f * h, -1, -1));

            if (nv >= 0) {
                if (promote_junction(nv, a, b, h)) {
                    ++nfix;
                } else {
                    ++sk_near;
                }

                continue;
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(a == 0 ? b : a));
            nd.typ.push_back(static_cast<uint8_t>(typ));
            nd.part.push_back(static_cast<uint16_t>(a == 0 ? 0 : b));
            nd.part.push_back(static_cast<uint16_t>(cc));
            nd.part3.push_back(V2M_NOLAB);
            touched.push_back(1);
        }

        ++nfix;
    }

#undef FLD

    if (std::getenv("V2M_REPAIR_DEBUG")) {
        std::fprintf(stderr, "[repair] %zu fixes: %zu applied (%zu junction, %zu face); skipped: %zu same-label, %zu no "
                     "outside sample, %zu no sign change, %zu near a node, %zu junction, %zu face\n", fixes.size(), nfix,
                     njf, nff, sk_same, sk_out, sk_sign, sk_near, sk_jf, sk_ff);
    }

    G.n = nd.size();

    // a moved node left a stale entry in its old cell: drop it, and put the cell it
    // went to back in index order -- the grid a fresh build would give, without
    // rebuilding the map of every node (~0.8 s at 2.7M nodes)
    for (const auto& mf : moved_from) {
        std::vector<uint32_t>& cv = grid[mf.second];
        const auto it = std::find(cv.begin(), cv.end(), mf.first);

        if (it != cv.end()) {
            cv.erase(it);
        }
    }

    for (const auto& mf : moved_from) {
        std::vector<uint32_t>& cv = grid[ckey(&nd.P[3 * mf.first])];
        std::sort(cv.begin(), cv.end());
    }

    return nfix;
}

// Quality-guarded ODT smoothing of the INTERIOR nodes with the connectivity
// kept: each proposes half a step toward the volume-weighted mean of its tets'
// circumcentres (clamped to 2 h, the step to 0.2 h); a proposal is taken only if
// every incident tet keeps its orientation, the worst incident Joe-Liu quality
// improves and the node stays inside its own label. Proposals conflict when two
// nodes share a tet: a node moves if no proposing neighbour has a smaller index
// (an independent set), so each pass is parallel. Interface / junction / corner
// nodes and the tet labels are untouched, so conformity is kept.
// Volume-preserving smoothing of the region surfaces (SurfSmoothParams; iso2mesh's
// smoothsurf). The surfaces: the faces between tets of different labels (or none);
// each node's role from the label pairs of its faces -- one pair: an interface
// node, smoothed with its neighbours on that interface; two or more: a junction
// node, smoothed along its curve (the edges whose faces carry 2+ pairs) when it has
// exactly two curve neighbours, else (a corner, a branch) fixed. Every pass is a
// Jacobi step (the previous positions), as smoothsurf's; a node whose new place
// would invert one of its tets (check_tets), or turn one of its interface
// triangles over, keeps its old place in that pass. Returns the node moves made.
static size_t smooth_surfaces(Nodes& nd, TetOut& m, const SurfSmoothParams& ss, bool check_tets, size_t& nsurf,
                              size_t& blocked) {
    const int n = static_cast<int>(nd.size());
    const int64_t nt = static_cast<int64_t>(m.label.size());
    nsurf = blocked = 0;

    if (ss.iters <= 0 || nt == 0) {
        return 0;
    }

    static const int F[4][3] = { { 1, 2, 3 }, { 0, 3, 2 }, { 0, 1, 3 }, { 0, 2, 1 } };
    struct Fc {
        int v[3];
        int32_t lab;
        int64_t t;
    };
    std::vector<Fc> fc(static_cast<size_t>(nt) * 4);

    for (int64_t t = 0; t < nt; ++t)
        for (int f = 0; f < 4; ++f) {
            Fc& x = fc[static_cast<size_t>(t) * 4 + f];

            for (int k = 0; k < 3; ++k) {
                x.v[k] = m.tets[4 * t + F[f][k]];
            }

            std::sort(x.v, x.v + 3);
            x.lab = m.label[t];
            x.t = t;
        }

    std::sort(fc.begin(), fc.end(), [](const Fc& a, const Fc& b) {
        return a.v[0] != b.v[0] ? a.v[0] < b.v[0] : a.v[1] != b.v[1] ? a.v[1] < b.v[1] : a.v[2] < b.v[2];
    });
    // the interface faces: (nodes, the label pair low * 65536 + high); oriented
    // from its first tet, for the flip test
    struct If {
        int v[3];
        int64_t pair;
    };
    std::vector<If> faces;

    for (size_t i = 0; i < fc.size();) {
        size_t j = i + 1;

        while (j < fc.size() && fc[j].v[0] == fc[i].v[0] && fc[j].v[1] == fc[i].v[1] && fc[j].v[2] == fc[i].v[2]) {
            ++j;
        }

        const int32_t la = fc[i].lab, lb = j - i >= 2 ? fc[i + 1].lab : 0;

        if (la != lb) {
            If x;
            // (the face as its first tet sees it: its own winding)
            const int64_t t = fc[i].t;
            int lf = 0;

            for (int f = 0; f < 4; ++f) {
                int w[3] = { m.tets[4 * t + F[f][0]], m.tets[4 * t + F[f][1]], m.tets[4 * t + F[f][2]] };
                std::sort(w, w + 3);

                if (w[0] == fc[i].v[0] && w[1] == fc[i].v[1] && w[2] == fc[i].v[2]) {
                    lf = f;
                }
            }

            for (int k = 0; k < 3; ++k) {
                x.v[k] = m.tets[4 * t + F[lf][k]];
            }

            x.pair = static_cast<int64_t>(std::min(la, lb)) * 65536 + std::max(la, lb);
            faces.push_back(x);
        }

        i = j;
    }

    // per node: its faces, its pairs
    std::vector<int> fo(static_cast<size_t>(n) + 1, 0);

    for (const If& x : faces)
        for (int k = 0; k < 3; ++k) {
            ++fo[static_cast<size_t>(x.v[k]) + 1];
        }

    for (int v = 0; v < n; ++v) {
        fo[static_cast<size_t>(v) + 1] += fo[static_cast<size_t>(v)];
    }

    std::vector<int> fl(static_cast<size_t>(fo[static_cast<size_t>(n)]));
    {
        std::vector<int> c(fo.begin(), fo.end() - 1);

        for (size_t i = 0; i < faces.size(); ++i)
            for (int k = 0; k < 3; ++k) {
                fl[static_cast<size_t>(c[static_cast<size_t>(faces[i].v[k])]++)] = static_cast<int>(i);
            }
    }
    // the neighbours each node smooths with
    std::vector<std::vector<int>> nb(static_cast<size_t>(n));
    std::vector<char> fixed(static_cast<size_t>(n), 1);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int v = 0; v < n; ++v) {
        if (fo[static_cast<size_t>(v)] == fo[static_cast<size_t>(v) + 1]) {
            continue;   // (not on a surface)
        }

        std::vector<int64_t> pairs;

        for (int k = fo[static_cast<size_t>(v)]; k < fo[static_cast<size_t>(v) + 1]; ++k) {
            pairs.push_back(faces[static_cast<size_t>(fl[static_cast<size_t>(k)])].pair);
        }

        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
        // the edges v-w of its faces, with the pairs of the faces on each
        std::vector<std::pair<int, int64_t>> ew;

        for (int k = fo[static_cast<size_t>(v)]; k < fo[static_cast<size_t>(v) + 1]; ++k) {
            const If& x = faces[static_cast<size_t>(fl[static_cast<size_t>(k)])];

            for (int c = 0; c < 3; ++c)
                if (x.v[c] != v) {
                    ew.push_back({ x.v[c], x.pair });
                }
        }

        std::sort(ew.begin(), ew.end());
        std::vector<int> mine;

        if (pairs.size() == 1) {   // an interface node: its interface neighbours
            for (size_t k = 0; k < ew.size(); ++k)
                if (k == 0 || ew[k].first != ew[k - 1].first) {
                    mine.push_back(ew[k].first);
                }
        } else {   // a junction node: the neighbours along edges of 2+ pairs (its curve)
            for (size_t k = 0; k < ew.size();) {
                size_t j = k;
                std::vector<int64_t> pp;

                while (j < ew.size() && ew[j].first == ew[k].first) {
                    pp.push_back(ew[j].second);
                    ++j;
                }

                std::sort(pp.begin(), pp.end());

                if (std::unique(pp.begin(), pp.end()) - pp.begin() >= 2) {
                    mine.push_back(ew[k].first);
                }

                k = j;
            }

            if (mine.size() != 2) {
                mine.clear();   // (a corner / a branch: fixed)
            }
        }

        if (!mine.empty()) {
            nb[static_cast<size_t>(v)] = mine;
            fixed[static_cast<size_t>(v)] = 0;
        }
    }

    // (the domain's cut faces: a region cut by the image's bounds ends in a flat face
    // on the bounding box -- its nodes have neighbours on one side only; they stay)
    {
        float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };

        for (int v = 0; v < n; ++v)
            for (int c = 0; c < 3; ++c) {
                lo[c] = std::min(lo[c], nd.P[3 * static_cast<size_t>(v) + c]);
                hi[c] = std::max(hi[c], nd.P[3 * static_cast<size_t>(v) + c]);
            }

        for (int v = 0; v < n; ++v)
            for (int c = 0; c < 3; ++c) {
                const float e = 1e-4f * (hi[c] - lo[c]), x = nd.P[3 * static_cast<size_t>(v) + c];

                if (x <= lo[c] + e || x >= hi[c] - e) {
                    fixed[static_cast<size_t>(v)] = 1;
                }
            }
    }

    for (int v = 0; v < n; ++v) {
        nsurf += !fixed[static_cast<size_t>(v)];
    }

    // the tets at each node (for the inversion test)
    std::vector<int> to(static_cast<size_t>(n) + 1, 0);

    for (int64_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            ++to[static_cast<size_t>(m.tets[4 * t + k]) + 1];
        }

    for (int v = 0; v < n; ++v) {
        to[static_cast<size_t>(v) + 1] += to[static_cast<size_t>(v)];
    }

    std::vector<int> tl(static_cast<size_t>(to[static_cast<size_t>(n)]));
    {
        std::vector<int> c(to.begin(), to.end() - 1);

        for (int64_t t = 0; t < nt; ++t)
            for (int k = 0; k < 4; ++k) {
                tl[static_cast<size_t>(c[static_cast<size_t>(m.tets[4 * t + k])]++)] = static_cast<int>(t);
            }
    }
    std::vector<double> X(nd.P.begin(), nd.P.end()), O = X;   // (current; original, for HC)
    auto vol6 = [](const double* a, const double* b, const double* c, const double* e) {
        const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, w[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] },
                     z[3] = { e[0] - a[0], e[1] - a[1], e[2] - a[2] };
        return u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0]) + u[2] * (w[0] * z[1] - w[1] * z[0]);
    };
    std::vector<double> sgn(static_cast<size_t>(nt));   // each tet's orientation, as tessellated

    for (int64_t t = 0; t < nt; ++t) {
        const int* q = &m.tets[4 * t];
        sgn[static_cast<size_t>(t)] = vol6(&X[3 * q[0]], &X[3 * q[1]], &X[3 * q[2]], &X[3 * q[3]]) >= 0 ? 1.0 : -1.0;
    }

    std::vector<double> fn0(faces.size() * 3);   // the interface triangles' normals, as tessellated

    for (size_t i = 0; i < faces.size(); ++i) {
        const double* a = &X[3 * faces[i].v[0]], *b = &X[3 * faces[i].v[1]], *c = &X[3 * faces[i].v[2]];
        const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, w[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        fn0[3 * i] = u[1] * w[2] - u[2] * w[1];
        fn0[3 * i + 1] = u[2] * w[0] - u[0] * w[2];
        fn0[3 * i + 2] = u[0] * w[1] - u[1] * w[0];
    }

    const double a = ss.alpha, ia = 1 - a;
    const bool hc = ss.method == "laplacianhc", lowpass = ss.method == "lowpass";
    auto mean_of = [&](const std::vector<double>& Q, int v, double* r) {
        r[0] = r[1] = r[2] = 0;

        for (const int w : nb[static_cast<size_t>(v)])
            for (int c = 0; c < 3; ++c) {
                r[c] += Q[3 * static_cast<size_t>(w) + c];
            }

        const double k = 1.0 / static_cast<double>(nb[static_cast<size_t>(v)].size());

        for (int c = 0; c < 3; ++c) {
            r[c] *= k;
        }
    };
    // accept the new places P of the moving nodes where no tet inverts and no
    // interface triangle turns over; the others keep Q's
    auto commit = [&](std::vector<double>& P, const std::vector<double>& Q) {
        size_t bad = 0;

        for (int round = 0; round < 8; ++round) {
            std::vector<char> undo(static_cast<size_t>(n), 0);
            size_t nb_ = 0;
            #pragma omp parallel for schedule(dynamic, 1024) reduction(+ : nb_)

            for (int v = 0; v < n; ++v) {
                if (fixed[static_cast<size_t>(v)] || (P[3 * v] == Q[3 * v] && P[3 * v + 1] == Q[3 * v + 1] && P[3 * v + 2] == Q[3 * v + 2])) {
                    continue;
                }

                bool ok = true;

                for (int k = to[static_cast<size_t>(v)]; check_tets && k < to[static_cast<size_t>(v) + 1] && ok; ++k) {
                    const int t = tl[static_cast<size_t>(k)];
                    const int* q = &m.tets[4 * static_cast<int64_t>(t)];
                    ok = vol6(&P[3 * q[0]], &P[3 * q[1]], &P[3 * q[2]], &P[3 * q[3]]) * sgn[static_cast<size_t>(t)] > 0;
                }

                for (int k = fo[static_cast<size_t>(v)]; k < fo[static_cast<size_t>(v) + 1] && ok; ++k) {
                    const size_t i = static_cast<size_t>(fl[static_cast<size_t>(k)]);
                    const double* pa = &P[3 * faces[i].v[0]], *pb = &P[3 * faces[i].v[1]], *pc = &P[3 * faces[i].v[2]];
                    const double u[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] }, w[3] = { pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2] };
                    const double nn[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0] };
                    ok = nn[0] * fn0[3 * i] + nn[1] * fn0[3 * i + 1] + nn[2] * fn0[3 * i + 2] > 0;
                }

                if (!ok) {
                    undo[static_cast<size_t>(v)] = 1;
                    ++nb_;
                }
            }

            if (nb_ == 0) {
                break;
            }

            bad += nb_;

            for (int v = 0; v < n; ++v)
                if (undo[static_cast<size_t>(v)])
                    for (int c = 0; c < 3; ++c) {
                        P[3 * static_cast<size_t>(v) + c] = Q[3 * static_cast<size_t>(v) + c];
                    }
        }

        return bad;
    };
    size_t moved = 0;
    std::vector<double> P(X), B(X.size(), 0.0);
    auto count = [&](const std::vector<double>& A, const std::vector<double>& Q) {
        for (int v = 0; v < n; ++v)
            if (!fixed[static_cast<size_t>(v)] && (A[3 * v] != Q[3 * v] || A[3 * v + 1] != Q[3 * v + 1] || A[3 * v + 2] != Q[3 * v + 2])) {
                ++moved;
            }
    };

    for (int it = 0; it < ss.iters; ++it) {
        if (hc) {   // Laplacian-HC (smoothsurf's 'laplacianhc'): O the original nodes
            const std::vector<double> Q = X;
            #pragma omp parallel for schedule(dynamic, 1024)

            for (int v = 0; v < n; ++v) {
                if (fixed[static_cast<size_t>(v)]) {
                    continue;
                }

                double mq[3];
                mean_of(Q, v, mq);

                for (int c = 0; c < 3; ++c) {
                    P[3 * static_cast<size_t>(v) + c] = mq[c];
                    B[3 * static_cast<size_t>(v) + c] = mq[c] - (a * O[3 * static_cast<size_t>(v) + c] + ia * Q[3 * static_cast<size_t>(v) + c]);
                }
            }

            #pragma omp parallel for schedule(dynamic, 1024)

            for (int v = 0; v < n; ++v) {
                if (fixed[static_cast<size_t>(v)]) {
                    continue;
                }

                double mb[3];
                mean_of(B, v, mb);

                for (int c = 0; c < 3; ++c) {
                    P[3 * static_cast<size_t>(v) + c] -= ss.beta * B[3 * static_cast<size_t>(v) + c] + (1 - ss.beta) * mb[c];
                }
            }

            blocked += commit(P, Q);
            count(P, Q);
        } else {   // Laplacian / Taubin's low-pass (+alpha, then -1.02 alpha)
            for (int step = 0; step < (lowpass ? 2 : 1); ++step) {
                const double w = step == 0 ? a : -1.02 * a;
                const std::vector<double> Q = X;
                #pragma omp parallel for schedule(dynamic, 1024)

                for (int v = 0; v < n; ++v) {
                    if (fixed[static_cast<size_t>(v)]) {
                        continue;
                    }

                    double mq[3];
                    mean_of(Q, v, mq);

                    for (int c = 0; c < 3; ++c) {
                        P[3 * static_cast<size_t>(v) + c] = (1 - w) * Q[3 * static_cast<size_t>(v) + c] + w * mq[c];
                    }
                }

                blocked += commit(P, Q);
                count(P, Q);
                X = P;
            }
        }

        X = P;
    }

    for (size_t i = 0; i < X.size(); ++i) {
        nd.P[i] = static_cast<float>(X[i]);
    }

    m.P = nd.P;
    return moved;
}

static size_t smooth_interior(const Grid& g, Nodes& nd, TetOut& m, int passes) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
    const int n = static_cast<int>(nd.size());
    const int64_t nt = static_cast<int64_t>(m.label.size());
    std::vector<int> cnt(static_cast<size_t>(n) + 1, 0);

    for (int64_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            ++cnt[m.tets[4 * t + k] + 1];
        }

    for (int i = 0; i < n; ++i) {
        cnt[i + 1] += cnt[i];
    }

    std::vector<int> cur(cnt.begin(), cnt.end() - 1), n2t(static_cast<size_t>(cnt[n]));

    for (int64_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            n2t[cur[m.tets[4 * t + k]]++] = static_cast<int>(t);
        }

    auto vol6 = [](const double* a, const double* b, const double* c, const double* e) {
        const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] },
                     w[3] = { e[0] - a[0], e[1] - a[1], e[2] - a[2] };
        return u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0]);
    };
    size_t moved = 0;
    std::vector<float> np(static_cast<size_t>(n) * 3);
    std::vector<char> prop(n);

    for (int pass = 0; pass < passes; ++pass) {
        std::fill(prop.begin(), prop.end(), 0);
        #pragma omp parallel for schedule(monotonic: dynamic, 1024)

        for (int v = 0; v < n; ++v) {
            if (nd.typ[v] != V2M_INTERIOR || cnt[v + 1] == cnt[v]) {
                continue;
            }

            const double xv[3] = { nd.P[3 * v], nd.P[3 * v + 1], nd.P[3 * v + 2] };
            const double h = v2m_h_at(d, g.h.data(), nd.P[3 * v], nd.P[3 * v + 1], nd.P[3 * v + 2]);
            double S = 0.0, C[3] = { 0, 0, 0 }, qold = 180.0;
            double cand[16][3];
            int nc = 0;

            for (int s2 = cnt[v]; s2 < cnt[v + 1]; ++s2) {
                const int t = n2t[s2];
                double q[4][3];
                const double* pp[4];
                int iv = 0;

                for (int k2 = 0; k2 < 4; ++k2) {
                    iv = m.tets[4 * t + k2] == v ? k2 : iv;

                    for (int c = 0; c < 3; ++c) {
                        q[k2][c] = nd.P[3 * m.tets[4 * t + k2] + c];
                    }

                    pp[k2] = q[k2];
                }

                double md, jl, vol, o[3];
                tet_quality(pp, md, jl, vol);
                qold = std::min(qold, md);

                if (md < 15.0 && nc + 3 <= 15) {
                    // sliver perturbation: along the normal of the opposite face, away
                    // from it (raising the sliver's height), three step sizes
                    int f[3], mf = 0;

                    for (int k2 = 0; k2 < 4; ++k2)
                        if (k2 != iv) {
                            f[mf++] = k2;
                        }

                    const double u[3] = { q[f[1]][0] - q[f[0]][0], q[f[1]][1] - q[f[0]][1], q[f[1]][2] - q[f[0]][2] },
                                 w[3] = { q[f[2]][0] - q[f[0]][0], q[f[2]][1] - q[f[0]][1], q[f[2]][2] - q[f[0]][2] };
                    double nn[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0] };
                    const double nl = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);

                    if (nl > 0.0) {
                        const double side = (xv[0] - q[f[0]][0]) * nn[0] + (xv[1] - q[f[0]][1]) * nn[1] + (xv[2] - q[f[0]][2]) * nn[2];
                        const double sg = side >= 0.0 ? 1.0 : -1.0;

                        for (const double a : { 0.1, 0.2, 0.35 }) {
                            for (int c = 0; c < 3; ++c) {
                                cand[nc][c] = xv[c] + sg * a * h * nn[c] / nl;
                            }

                            ++nc;
                        }
                    }
                }

                if (!circumcentre(pp[0], pp[1], pp[2], pp[3], o)) {
                    continue;
                }

                double e[3] = { o[0] - xv[0], o[1] - xv[1], o[2] - xv[2] };
                const double el = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);

                if (el > 2.0 * h) {   // a sliver's circumcentre can be far away
                    for (int k2 = 0; k2 < 3; ++k2) {
                        e[k2] *= 2.0 * h / el;
                    }
                }

                S += vol;

                for (int k2 = 0; k2 < 3; ++k2) {
                    C[k2] += vol * (xv[k2] + e[k2]);
                }
            }

            if (S > 0.0) {   // the ODT candidate: half a step, capped at 0.2 h
                double dl[3] = { 0.5 * (C[0] / S - xv[0]), 0.5 * (C[1] / S - xv[1]), 0.5 * (C[2] / S - xv[2]) };
                const double dn = std::sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);

                if (dn > 1e-4 * h) {
                    const double sc = dn > 0.2 * h ? 0.2 * h / dn : 1.0;

                    for (int c = 0; c < 3; ++c) {
                        cand[nc][c] = xv[c] + sc * dl[c];
                    }

                    ++nc;
                }
            }

            // the best candidate by the star's minimum dihedral
            double best = qold + 0.1;
            int bi = -1;

            for (int ci = 0; ci < nc; ++ci) {
                const double* xn = cand[ci];
                int sec;
                float mg;

                if (v2m_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI,
                                g.gTW.data(), g.gm, 0, static_cast<float>(xn[0]), static_cast<float>(xn[1]),
                                static_cast<float>(xn[2]), &sec, &mg) != nd.lab[v]) {
                    continue;
                }

                double qnew = 180.0;
                bool ok = true;

                for (int s2 = cnt[v]; s2 < cnt[v + 1] && ok && qnew > best; ++s2) {
                    const int t = n2t[s2];
                    double q[4][3], r[4][3];
                    const double* pp[4];

                    for (int k2 = 0; k2 < 4; ++k2) {
                        const int w = m.tets[4 * t + k2];

                        for (int c = 0; c < 3; ++c) {
                            q[k2][c] = nd.P[3 * w + c];
                            r[k2][c] = w == v ? xn[c] : q[k2][c];
                        }

                        pp[k2] = r[k2];
                    }

                    if ((vol6(q[0], q[1], q[2], q[3]) > 0.0) != (vol6(r[0], r[1], r[2], r[3]) > 0.0)) {
                        ok = false;   // would invert
                        break;
                    }

                    double md, jl, vol;
                    tet_quality(pp, md, jl, vol);
                    qnew = std::min(qnew, md);
                }

                if (ok && qnew > best) {
                    best = qnew;
                    bi = ci;
                }
            }

            if (bi >= 0) {
                prop[v] = 1;

                for (int c = 0; c < 3; ++c) {
                    np[3 * v + c] = static_cast<float>(cand[bi][c]);
                }
            }
        }

        // independent set: v moves if no proposing node sharing a tet has a smaller index
        size_t mv = 0;
        std::vector<char> go(n, 0);
        #pragma omp parallel for schedule(monotonic: dynamic, 1024) reduction(+ : mv)

        for (int v = 0; v < n; ++v) {
            if (!prop[v]) {
                continue;
            }

            bool win = true;

            for (int s2 = cnt[v]; s2 < cnt[v + 1] && win; ++s2) {
                const int t = n2t[s2];

                for (int k = 0; k < 4; ++k) {
                    const int w = m.tets[4 * t + k];

                    if (w < v && prop[w]) {
                        win = false;
                        break;
                    }
                }
            }

            go[v] = win ? 1 : 0;
            mv += win;
        }

        #pragma omp parallel for

        for (int v = 0; v < n; ++v)
            if (go[v]) {
                for (int k = 0; k < 3; ++k) {
                    nd.P[3 * v + k] = np[3 * v + k];
                }
            }

        moved += mv;

        if (mv == 0) {
            break;
        }
    }

    m.P = nd.P;
    return moved;
}

// Deviation metrics of the final mesh: the distance (voxels) of each offending
// node to the voxel interface it should lie on -- for the bad faces (nodes not on
// the face's a|b interface) and the spanning tets (nodes lacking the tet's label)
static void deviation_metrics(const Grid& g, const Nodes& nd, const std::vector<Fix>& facefixes,
                              const std::vector<std::array<uint32_t, 5>>& span_tets, TetStats& st) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
    const float vm = std::min(g.vs[0], std::min(g.vs[1], g.vs[2]));
    // exact, field-independent: the distance to the nearest face between an l1
    // voxel and an l2 voxel (either orientation), searched within 4 voxels
    auto dist = [&](int l1, int l2, const float* p) {
        float y[3];
        float d2 = v2m_vox_nearest(d, g.L->data(), l1, l2, V2M_NOLAB, p, 4, y);
        const float e2 = v2m_vox_nearest(d, g.L->data(), l2, l1, V2M_NOLAB, p, 4, y);

        if (d2 < 0.0f || (e2 >= 0.0f && e2 < d2)) {
            d2 = e2;
        }

        return d2 < 0.0f ? 5.0f : std::sqrt(d2) / vm;
    };
    auto pct = [](std::vector<float>& x, double* out) {
        for (int k = 0; k < 4; ++k) {
            out[k] = 0.0;
        }

        if (x.empty()) {
            return;
        }

        std::sort(x.begin(), x.end());
        out[0] = x[x.size() / 2];
        out[1] = x[static_cast<size_t>(0.95 * (x.size() - 1))];
        out[2] = x[static_cast<size_t>(0.99 * (x.size() - 1))];
        out[3] = x.back();
    };
    std::vector<float> df(facefixes.size(), 0.0f), ds(span_tets.size(), 0.0f);
    #pragma omp parallel for schedule(monotonic: dynamic, 256)

    for (int64_t i = 0; i < static_cast<int64_t>(facefixes.size()); ++i) {
        const Fix& f = facefixes[i];

        for (int k = 0; k < 3; ++k) {
            int S[4];
            const int n = node_label_set(nd, f.tv[k], S);
            bool ha = false, hb = false;

            for (int e = 0; e < n; ++e) {
                ha |= S[e] == f.a;
                hb |= S[e] == f.b;
            }

            if (nd.typ[f.tv[k]] == V2M_INTERIOR || !ha || !hb) {
                df[i] = std::max(df[i], dist(f.a, f.b, &nd.P[3 * f.tv[k]]));
            }
        }
    }

    #pragma omp parallel for schedule(monotonic: dynamic, 256)

    for (int64_t i = 0; i < static_cast<int64_t>(span_tets.size()); ++i) {
        const int lt = static_cast<int>(span_tets[i][4]);

        for (int k = 0; k < 4; ++k) {
            int S[4];
            const int n = node_label_set(nd, span_tets[i][k], S);
            bool has = false;
            float dm = 99.0f;

            for (int e = 0; e < n; ++e) {
                has |= S[e] == lt;
            }

            if (has) {
                continue;
            }

            for (int e = 0; e < n; ++e) {
                dm = std::min(dm, dist(S[e], lt, &nd.P[3 * span_tets[i][k]]));
            }

            ds[i] = std::max(ds[i], dm);
        }
    }

    pct(df, st.dev_face);
    pct(ds, st.dev_span);
}

// Radius-edge quality refinement (TetGen / gpu_brain2mesh -q): a kept tet whose
// circumradius / shortest edge exceeds q gets a node at its circumcentre -- an
// INTERIOR node of the tet's label if the circumcentre lies inside that label,
// else the circumcentre projected (bracketed, within 0.5 h) onto the interface
// between the tet's label and the one it falls in, as an INTERFACE node (the
// Ruppert / Shewchuk rule: a point that would encroach the boundary goes on it).
// Skipped: tets already small for the sizing field (circumradius < 0.3 h: bad by
// a short edge, splitting would only make more), and points within 0.3 h of a
// node. Returns the number of nodes added.
static size_t quality_refine(const Grid& g, const TetOut& m, Nodes& nd, double q) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
#define QFLD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm
    struct Cand {
        float c[3];
        int lab;
        double rr;
        int other;   // the other label on the tet's nodes, -1 if none
    };
    const int64_t nt = static_cast<int64_t>(m.label.size());
    std::vector<std::vector<Cand>> per(omp_get_max_threads());
    #pragma omp parallel
    {
        std::vector<Cand>& mine = per[omp_get_thread_num()];
        #pragma omp for schedule(monotonic: dynamic, 4096)

        for (int64_t t = 0; t < nt; ++t) {
            double p[4][3];

            for (int k = 0; k < 4; ++k)
                for (int e = 0; e < 3; ++e) {
                    p[k][e] = nd.P[3 * m.tets[4 * t + k] + e];
                }

            double o[3];

            if (!circumcentre(p[0], p[1], p[2], p[3], o)) {
                continue;
            }

            double emin = 1e300;
            const int ei[6][2] = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };

            for (int k = 0; k < 6; ++k) {
                const double dx = p[ei[k][0]][0] - p[ei[k][1]][0], dy = p[ei[k][0]][1] - p[ei[k][1]][1],
                             dz = p[ei[k][0]][2] - p[ei[k][1]][2];
                emin = std::min(emin, dx * dx + dy * dy + dz * dz);
            }

            const double R = std::sqrt((o[0] - p[0][0]) * (o[0] - p[0][0]) + (o[1] - p[0][1]) * (o[1] - p[0][1]) +
                                       (o[2] - p[0][2]) * (o[2] - p[0][2]));

            if (R <= q * std::sqrt(emin)) {
                continue;
            }

            const float c[3] = { static_cast<float>(o[0]), static_cast<float>(o[1]), static_cast<float>(o[2]) };

            if (c[0] < 0.0f || c[1] < 0.0f || c[2] < 0.0f || c[0] > (g.nx - 1) * g.vs[0] || c[1] > (g.ny - 1) * g.vs[1] ||
                    c[2] > (g.nz - 1) * g.vs[2]) {
                continue;
            }

            if (R < 0.3 * v2m_h_at(d, g.h.data(), c[0], c[1], c[2])) {
                continue;
            }

            // conservative: only where at most two labels meet and no junction /
            // corner node is involved (splitting a tet in a thin layer / at a triple
            // line created configurations the conformity repairs could not undo)
            int U[8], nu = 0;
            bool jn = false;

            for (int k = 0; k < 4 && !jn; ++k) {
                const uint32_t v = static_cast<uint32_t>(m.tets[4 * t + k]);
                int S[4];
                const int ns = node_label_set(nd, v, S);
                jn = nd.typ[v] >= V2M_JUNCTION;

                for (int e = 0; e < ns; ++e) {
                    bool seen = false;

                    for (int x = 0; x < nu; ++x) {
                        seen |= U[x] == S[e];
                    }

                    if (!seen && nu < 8) {
                        U[nu++] = S[e];
                    }
                }
            }

            if (jn || nu > 2) {
                continue;
            }

            Cand cd = { { c[0], c[1], c[2] }, m.label[t], R / std::sqrt(emin), -1 };

            for (int x = 0; x < nu; ++x)
                if (U[x] != m.label[t]) {
                    cd.other = U[x];   // the one interface this tet touches (if any)
                }

            mine.push_back(cd);
        }
    }

    std::vector<Cand> cand;

    for (auto& v : per) {
        cand.insert(cand.end(), v.begin(), v.end());
    }

    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {   // worst first, deterministic
        return a.rr > b.rr || (a.rr == b.rr && (a.c[0] < b.c[0] || (a.c[0] == b.c[0] && (a.c[1] < b.c[1] ||
                                                (a.c[1] == b.c[1] && a.c[2] < b.c[2])))));
    });
    // spacing hash (cell 0.3 hmin) over all nodes
    const float cell = 0.3f * g.hmin;
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    auto ckey = [&](const float* x) {
        const int64_t i = static_cast<int64_t>(std::floor(x[0] / cell)), j = static_cast<int64_t>(std::floor(x[1] / cell)),
                      k = static_cast<int64_t>(std::floor(x[2] / cell));
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };

    for (uint32_t v = 0; v < nd.size(); ++v) {
        grid[ckey(&nd.P[3 * v])].push_back(v);
    }

    auto near_node = [&](const float* x, float r) {
        const int mm = static_cast<int>(std::ceil(r / cell));

        for (int dz = -mm; dz <= mm; ++dz)
            for (int dy = -mm; dy <= mm; ++dy)
                for (int dx = -mm; dx <= mm; ++dx) {
                    const float y[3] = { x[0] + dx * cell, x[1] + dy * cell, x[2] + dz * cell };
                    auto it = grid.find(ckey(y));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (uint32_t v : it->second) {
                        const float ex = nd.P[3 * v] - x[0], ey = nd.P[3 * v + 1] - x[1], ez = nd.P[3 * v + 2] - x[2];

                        if (ex * ex + ey * ey + ez * ez < r * r) {
                            return true;
                        }
                    }
                }

        return false;
    };
    size_t added = 0;

    for (const Cand& k : cand) {
        float c[3] = { k.c[0], k.c[1], k.c[2] };
        const float h = v2m_h_at(d, g.h.data(), c[0], c[1], c[2]);
        int sec;
        float mg;
        const int lc = v2m_label_of(QFLD, 0, c[0], c[1], c[2], &sec, &mg);
        int typ = V2M_INTERIOR, own = k.lab, part = V2M_NOLAB;

        if (lc != k.lab) {   // encroaches: onto the k.lab | lc interface -- one the tet touches
            if (lc != k.other) {
                continue;
            }

            own = k.lab;
            part = lc;

            if (!v2m_project1(QFLD, own, part, c, 0.5f * h) || !v2m_valid_on(QFLD, own, part, V2M_NOLAB, c)) {
                continue;
            }

            typ = V2M_INTERFACE;
        } else if (sec != V2M_NOLAB && mg < 0.1f) {   // on the interface already, practically
            continue;
        }

        if (near_node(c, 0.3f * h)) {
            continue;
        }

        grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
        nd.P.insert(nd.P.end(), c, c + 3);
        nd.lab.push_back(static_cast<uint16_t>(own));
        nd.typ.push_back(static_cast<uint8_t>(typ));
        nd.part.push_back(static_cast<uint16_t>(part));
        nd.part.push_back(V2M_NOLAB);
        nd.part3.push_back(V2M_NOLAB);
        ++added;
    }

#undef QFLD
    return added;
}

// Pre-snap (before the first Delaunay): an interior node within 0.3 h of an
// interface -- found by the bracketed projection, robust where |grad psi| is
// small (the relaxation's snap test estimates the distance as psi / |grad psi|,
// which the sharp thin-layer field defeats) -- is put on it. These nodes used to
// be promoted by the first repair round instead, and a moved node forced a full
// Delaunay rebuild (the live one cannot delete a vertex). Returns the count.
static size_t presnap_interior(const Grid& g, Nodes& nd) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
#define PFLD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm
    const int n = static_cast<int>(nd.size());
    std::vector<float> np(static_cast<size_t>(n) * 3);
    std::vector<int> ok(n, -1);   // partner label if snapped
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int v = 0; v < n; ++v) {
        if (nd.typ[v] != V2M_INTERIOR) {
            continue;
        }

        float q[3] = { nd.P[3 * v], nd.P[3 * v + 1], nd.P[3 * v + 2] };
        int sec;
        float mg;
        const int l = v2m_label_of(PFLD, 0, q[0], q[1], q[2], &sec, &mg);

        if (l != nd.lab[v] || sec == V2M_NOLAB) {
            continue;
        }

        const float h = v2m_h_at(d, g.h.data(), q[0], q[1], q[2]);

        if (v2m_project1(PFLD, nd.lab[v], sec, q, 0.3f * h) && v2m_valid_on(PFLD, nd.lab[v], sec, V2M_NOLAB, q) &&
                v2m_third_label(PFLD, nd.lab[v], sec, q) == V2M_NOLAB) {
            ok[v] = sec;

            for (int k = 0; k < 3; ++k) {
                np[3 * v + k] = q[k];
            }
        }
    }

    // spacing: a snapped position must keep 0.3 h from every other node (serial,
    // in index order: deterministic)
    const float cell = 0.3f * g.hmin;
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    auto ckey = [&](const float* x) {
        const int64_t i = static_cast<int64_t>(std::floor(x[0] / cell)), j = static_cast<int64_t>(std::floor(x[1] / cell)),
                      k = static_cast<int64_t>(std::floor(x[2] / cell));
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };

    for (uint32_t v = 0; v < static_cast<uint32_t>(n); ++v) {
        grid[ckey(&nd.P[3 * v])].push_back(v);
    }

    size_t snapped = 0;

    for (int v = 0; v < n; ++v) {
        if (ok[v] < 0) {
            continue;
        }

        const float* x = &np[3 * v];
        const float h = v2m_h_at(d, g.h.data(), x[0], x[1], x[2]);
        const float r = 0.3f * h;
        const int mm = static_cast<int>(std::ceil(r / cell));
        bool near = false;

        for (int dz = -mm; dz <= mm && !near; ++dz)
            for (int dy = -mm; dy <= mm && !near; ++dy)
                for (int dx = -mm; dx <= mm && !near; ++dx) {
                    const float y[3] = { x[0] + dx * cell, x[1] + dy * cell, x[2] + dz * cell };
                    auto it = grid.find(ckey(y));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (uint32_t w : it->second) {
                        if (static_cast<int>(w) == v) {
                            continue;
                        }

                        const float ex = nd.P[3 * w] - x[0], ey = nd.P[3 * w + 1] - x[1], ez = nd.P[3 * w + 2] - x[2];

                        if (ex * ex + ey * ey + ez * ez < r * r) {
                            near = true;
                            break;
                        }
                    }
                }

        if (near) {
            continue;
        }

        auto& old = grid[ckey(&nd.P[3 * v])];   // move v in the spacing grid
        old.erase(std::find(old.begin(), old.end(), static_cast<uint32_t>(v)));

        for (int k = 0; k < 3; ++k) {
            nd.P[3 * v + k] = x[k];
        }

        grid[ckey(x)].push_back(static_cast<uint32_t>(v));
        nd.typ[v] = V2M_INTERFACE;
        nd.part[2 * v] = static_cast<uint16_t>(ok[v]);
        nd.part[2 * v + 1] = V2M_NOLAB;
        ++snapped;
    }

#undef PFLD
    return snapped;
}

// Drop the nodes that sit exactly on another node (keeping the first): the exact
// Delaunay's symbolic perturbation assumes distinct points and aborts otherwise.
// The relaxation can project two junction / corner nodes onto the same point
// (seen once in 1.2M nodes on a 17-label TPM); the repairs keep their own spacing.
static size_t remove_coincident(Nodes& nd) {
    const size_t n = nd.size();
    std::unordered_map<uint64_t, uint32_t> seen;
    seen.reserve(n * 2);
    std::vector<char> drop(n, 0);
    size_t nd_drop = 0;

    for (uint32_t v = 0; v < n; ++v) {
        uint32_t b[3];
        std::memcpy(b, &nd.P[3 * v], 12);
        const uint64_t h = (static_cast<uint64_t>(b[0]) * 0x9E3779B97F4A7C15ULL) ^
                           (static_cast<uint64_t>(b[1]) * 0xC2B2AE3D27D4EB4FULL) ^ (static_cast<uint64_t>(b[2]) * 0x165667B19E3779F9ULL);
        auto it = seen.find(h);

        if (it != seen.end() && std::memcmp(&nd.P[3 * it->second], &nd.P[3 * v], 12) == 0) {
            drop[v] = 1;
            ++nd_drop;
        } else if (it == seen.end()) {
            seen.emplace(h, v);
        }
    }

    if (!nd_drop) {
        return 0;
    }

    const bool p3 = nd.part3.size() == n;
    size_t w = 0;

    for (size_t v = 0; v < n; ++v) {
        if (drop[v]) {
            continue;
        }

        for (int k = 0; k < 3; ++k) {
            nd.P[3 * w + k] = nd.P[3 * v + k];
        }

        nd.lab[w] = nd.lab[v];
        nd.typ[w] = nd.typ[v];
        nd.part[2 * w] = nd.part[2 * v];
        nd.part[2 * w + 1] = nd.part[2 * v + 1];

        if (p3) {
            nd.part3[w] = nd.part3[v];
        }

        ++w;
    }

    nd.P.resize(3 * w);
    nd.lab.resize(w);
    nd.typ.resize(w);
    nd.part.resize(2 * w);

    if (p3) {
        nd.part3.resize(w);
    }

    return nd_drop;
}

// Every node must lie where the field's argmax is one of its labels: the
// conformity checks and the repair walks use the strict argmax, while a node is
// accepted during the relaxation within a 0.02 tolerance (v2m_valid_on). With
// label indicators the fields are sharp and such nodes are rare; with overlapping
// probabilities (--tpm-fields) a junction / interface node can sit on its a|b(|c)
// zero set where a third label dominates, and no repair of its edges succeeds.
// An invalid node is re-seated on the interface of the two top labels at its
// position (a short bracketed projection, kept if then strictly valid), else
// becomes an interior node of the top label, else (the exterior on top) is dropped.
static size_t validate_nodes(const Grid& g, Nodes& nd, size_t* reseated, size_t* dropped) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
#define VFLD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm
    const size_t n = nd.size();
    std::vector<char> drop(n, 0);
    size_t nbad = 0, nres = 0, ndrop = 0;
    auto top = [&](const float* x, int* l1, int* l2) {
        float mg;
        *l1 = v2m_label_of(VFLD, 0, x[0], x[1], x[2], l2, &mg);
    };

    for (uint32_t v = 0; v < n; ++v) {
        if (nd.typ[v] == V2M_INTERIOR) {
            continue;   // (an interior node never leaves its label: v2m_move traps it)
        }

        int S[4], l1, l2;
        const int ns = node_label_set(nd, v, S);
        top(&nd.P[3 * v], &l1, &l2);
        bool in = false;

        for (int k = 0; k < ns; ++k) {
            in = in || S[k] == l1;
        }

        if (in) {
            continue;
        }

        ++nbad;
        float q[3] = { nd.P[3 * v], nd.P[3 * v + 1], nd.P[3 * v + 2] };
        const float h = v2m_h_at(d, g.h.data(), q[0], q[1], q[2]);
        const int a = l1 != 0 ? l1 : l2, b = l1 != 0 ? l2 : l1;   // the own label is never 0

        if (a != V2M_NOLAB && b != V2M_NOLAB && a != 0 && v2m_project1(VFLD, a, b, q, 0.5f * h)) {
            int m1, m2;
            top(q, &m1, &m2);

            if (m1 == a || m1 == b) {
                nd.P[3 * v] = q[0];
                nd.P[3 * v + 1] = q[1];
                nd.P[3 * v + 2] = q[2];
                nd.lab[v] = static_cast<uint16_t>(a);
                nd.typ[v] = V2M_INTERFACE;
                nd.part[2 * v] = static_cast<uint16_t>(b);
                nd.part[2 * v + 1] = V2M_NOLAB;

                if (nd.part3.size() == n) {
                    nd.part3[v] = V2M_NOLAB;
                }

                ++nres;
                continue;
            }
        }

        if (l1 != 0) {   // inside label l1: an interior node there
            nd.lab[v] = static_cast<uint16_t>(l1);
            nd.typ[v] = V2M_INTERIOR;
            nd.part[2 * v] = V2M_NOLAB;
            nd.part[2 * v + 1] = V2M_NOLAB;

            if (nd.part3.size() == n) {
                nd.part3[v] = V2M_NOLAB;
            }
        } else {
            drop[v] = 1;
            ++ndrop;
        }
    }
#undef VFLD

    if (ndrop) {
        const bool p3 = nd.part3.size() == n;
        size_t w = 0;

        for (size_t v = 0; v < n; ++v) {
            if (drop[v]) {
                continue;
            }

            for (int k = 0; k < 3; ++k) {
                nd.P[3 * w + k] = nd.P[3 * v + k];
            }

            nd.lab[w] = nd.lab[v];
            nd.typ[w] = nd.typ[v];
            nd.part[2 * w] = nd.part[2 * v];
            nd.part[2 * w + 1] = nd.part[2 * v + 1];

            if (p3) {
                nd.part3[w] = nd.part3[v];
            }

            ++w;
        }

        nd.P.resize(3 * w);
        nd.lab.resize(w);
        nd.typ.resize(w);
        nd.part.resize(2 * w);

        if (p3) {
            nd.part3.resize(w);
        }
    }

    *reseated = nres;
    *dropped = ndrop;
    return nbad;
}

// Near-coincident nodes (closer than 0.2 h; remove_coincident only drops exact
// duplicates) make degenerate tets around them. The relaxation keeps interior and
// interface nodes apart, but junction nodes are fixed: with probability fields a
// node gliding onto a triple line can land on a junction seed, and two seeds from
// neighbouring cells can project to the same point (siamize SPM6: 4 junction pairs
// within 0.05 h, the worst slivers). Deterministic: corners, then junctions (lower
// index first) are kept unless within 0.2 h of a kept one; an interface node
// within 0.2 h of a kept junction / corner is dropped. Only the ~2% junction /
// corner nodes are filed in the grid.
static size_t merge_close_nodes(const Grid& g, Nodes& nd) {
    V2mDims d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.nbx = g.nbx;
    d.nby = g.nby;
    d.nbz = g.nbz;
    d.vx = g.vs[0];
    d.vy = g.vs[1];
    d.vz = g.vs[2];
    const size_t n = nd.size();
    const float frac = 0.2f, cell = frac * g.hmax;   // one cell covers the largest radius
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    auto key = [&](const float* x, int dx, int dy, int dz) {
        const int64_t i = static_cast<int64_t>(std::floor(x[0] / cell)) + dx, j = static_cast<int64_t>(std::floor(x[1] / cell)) + dy,
                      k = static_cast<int64_t>(std::floor(x[2] / cell)) + dz;
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };
    auto near_kept = [&](const float* x) {
        const float r = frac * v2m_h_at(d, g.h.data(), x[0], x[1], x[2]), r2 = r * r;

        for (int a = -1; a <= 1; ++a)
            for (int b = -1; b <= 1; ++b)
                for (int e = -1; e <= 1; ++e) {
                    auto it = grid.find(key(x, a, b, e));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (uint32_t u : it->second) {
                        const float* y = &nd.P[3 * u];
                        const float dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];

                        if (dx * dx + dy * dy + dz * dz < r2) {
                            return true;
                        }
                    }
                }

        return false;
    };
    std::vector<char> drop(n, 0);
    size_t ndrop = 0;

    for (int ty = V2M_CORNER; ty >= V2M_JUNCTION; --ty)
        for (uint32_t v = 0; v < n; ++v) {
            if (nd.typ[v] != ty) {
                continue;
            }

            if (near_kept(&nd.P[3 * v])) {
                drop[v] = 1;
                ++ndrop;
            } else {
                grid[key(&nd.P[3 * v], 0, 0, 0)].push_back(v);
            }
        }

    if (!grid.empty()) {
        #pragma omp parallel for schedule(static) reduction(+ : ndrop)

        for (int64_t v = 0; v < static_cast<int64_t>(n); ++v)
            if (nd.typ[v] == V2M_INTERFACE && near_kept(&nd.P[3 * v])) {
                drop[v] = 1;
                ++ndrop;
            }
    }

    if (!ndrop) {
        return 0;
    }

    const bool p3 = nd.part3.size() == n;
    size_t w = 0;

    for (size_t v = 0; v < n; ++v) {
        if (drop[v]) {
            continue;
        }

        for (int k = 0; k < 3; ++k) {
            nd.P[3 * w + k] = nd.P[3 * v + k];
        }

        nd.lab[w] = nd.lab[v];
        nd.typ[w] = nd.typ[v];
        nd.part[2 * w] = nd.part[2 * v];
        nd.part[2 * w + 1] = nd.part[2 * v + 1];

        if (p3) {
            nd.part3[w] = nd.part3[v];
        }

        ++w;
    }

    nd.P.resize(3 * w);
    nd.lab.resize(w);
    nd.typ.resize(w);
    nd.part.resize(2 * w);

    if (p3) {
        nd.part3.resize(w);
    }

    return ndrop;
}

// quality + per-label volumes over the kept tets (once, after the repairs)
static void mesh_quality(const Grid& g, const Nodes& nd, const TetOut& m, TetStats& st) {
    // 4. quality
    st.label_vol.assign(g.nlab, 0.0);
    st.label_vox.assign(g.nlab, 0.0);

    for (uint16_t l : *g.L) {
        if (l < st.label_vox.size()) {
            st.label_vox[l] += 1.0;
        }
    }

    for (double& x : st.label_vox) {
        x *= static_cast<double>(g.vs[0]) * g.vs[1] * g.vs[2];
    }

    // per tet in parallel, then accumulated in tet order (the sums as before);
    // the order statistics by nth_element instead of a full sort
    const int64_t nt = static_cast<int64_t>(m.label.size());
    std::vector<double> jl(nt), mdv(nt), vv(nt);
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t) {
        double q[4][3];
        const double* pp[4];

        for (int k = 0; k < 4; ++k) {
            for (int c = 0; c < 3; ++c) {
                q[k][c] = nd.P[3 * m.tets[4 * t + k] + c];
            }

            pp[k] = q[k];
        }

        tet_quality(pp, mdv[t], jl[t], vv[t]);

        if (mdv[t] < 10.0) {   // where the slivers are: by how many of their nodes are interior
            int ni = 0;

            for (int k = 0; k < 4; ++k) {
                ni += nd.typ[m.tets[4 * t + k]] == V2M_INTERIOR;
            }

            #pragma omp atomic
            ++st.sliver_by_interior[ni];
        }
    }

    double mind = 180.0, vol = 0.0;
    size_t s10 = 0, s5 = 0;

    for (int64_t t = 0; t < nt; ++t) {
        if (m.label[t] >= static_cast<int>(st.label_vol.size())) {
            st.label_vol.resize(m.label[t] + 1, 0.0);
        }

        st.label_vol[m.label[t]] += std::fabs(vv[t]);
        mind = std::min(mind, mdv[t]);
        s10 += mdv[t] < 10.0;
        s5 += mdv[t] < 5.0;
        vol += vv[t];
    }

    st.min_dihedral = mind;
    st.slivers10 = s10;
    st.slivers5 = s5;
    st.volume = vol;

    if (!jl.empty()) {
        const size_t n = jl.size();
        st.joe_liu_min = *std::min_element(jl.begin(), jl.end());
        std::nth_element(jl.begin(), jl.begin() + n / 20, jl.end());
        st.joe_liu_p5 = jl[n / 20];
        std::nth_element(jl.begin() + n / 20, jl.begin() + n / 2, jl.end());
        st.joe_liu_med = jl[n / 2];
    }

}

void tessellate(const Grid& g, Nodes& nd, bool voxel_mode, int max_repair, TetOut& m, TetStats& st, int smooth,
                bool opt, double q, bool surface_only, const SurfSmoothParams* ss) {
    // put near-interface interior nodes on the interface before the first Delaunay,
    // so the repairs only ever ADD nodes and every round stays incremental
    // (V2M_PROMOTE=1 restores the in-repair promotions)
    static const bool promote = std::getenv("V2M_PROMOTE") && std::atoi(std::getenv("V2M_PROMOTE")) > 0;
    // probability fields: interior nodes left just off an interface (the pre-snap
    // keeps 0.3 h from its dense interface nodes) blocked most crossing repairs
    // (siamize SPM6: ~1500 per round, labels ~50); the first repair round promotes
    // such an endpoint onto the crossing (one full Delaunay rebuild, ~2 s), spanning
    // tets 1974 -> 1752. A second promoting round gained nothing.
    const bool promote_first = g.prob_fields;
    // V2M_BATCH_MOVES=1: the junction promotions held back until the insertions run
    // dry (see repair_loop) -- on Colin27's TPM 10 -> 3 full rebuilds (tess 86 ->
    // 44 s) but ~33% more non-conforming faces left (2307 -> 3059): each move
    // round's follow-up repairs were what conformity needed, so off by default
    static const bool eager_moves = !(std::getenv("V2M_BATCH_MOVES") && std::atoi(std::getenv("V2M_BATCH_MOVES")) > 0);
    // V2M_TESS_TIMING: the time of each phase of the driver
    const bool ptiming = std::getenv("V2M_TESS_TIMING") != nullptr;
    clk::time_point tph = clk::now();
    auto phase = [&](const char* what) {
        if (ptiming) {
            std::fprintf(stderr, "[tp] %-16s %8.0f ms\n", what, since(tph));
            tph = clk::now();
        }
    };
    st.presnapped = promote ? 0 : presnap_interior(g, nd);
    phase("presnap");
    st.coincident = remove_coincident(nd);
    st.coincident += merge_close_nodes(g, nd);
    phase("coincident");
    if (g.prob_fields) {   // (see validate_nodes: overlapping probability fields)
        size_t nres = 0, ndrop = 0;
        const size_t nbad = validate_nodes(g, nd, &nres, &ndrop);

        if (std::getenv("V2M_TESS_VERBOSE")) {
            std::fprintf(stderr, "[valid] %zu nodes off their labels: %zu re-seated, %zu dropped, %zu made interior\n", nbad,
                         nres, ndrop, nbad - nres - ndrop);
        }
    }
    std::vector<Fix> fixes, ffix;
    std::vector<std::array<uint32_t, 5>> span_tets;
    std::vector<std::pair<int, int>> eout_prev;
    NodeGrid ngrid;
    int first_new = -1;   // first node added since the last round (insert-only rounds)
    std::unique_ptr<::TetMesh> live;
    bool rebuild = true;
    st.repair_rounds = 0;
    st.repaired = 0;

    // conformity repair rounds (crossings, junctions, exterior chords, faces) on the
    // live Delaunay, from the current node set until nothing is left to fix
    // V2M_Q_AT=k: the -q refinement as round k of the conformity repairs (its nodes
    // inserted into the live Delaunay, the repairs that follow fixing what they
    // broke) instead of a second repair loop after them; -1 (default): after
    static const int q_at = std::getenv("V2M_Q_AT") ? std::atoi(std::getenv("V2M_Q_AT")) : -1;
    const bool q_inline = q > 0.0 && q_at >= 0;
    st.q_added = 0;

    auto repair_loop = [&](bool allow_promote) {
        const bool qhere = allow_promote && q_inline;
        const int budget = max_repair + (qhere ? 1 : 0);   // (the -q round is not a repair round)

        for (int r = 0;; ++r) {
            tessellate_once(g, nd, voxel_mode, m, st, live, rebuild, ffix, fixes, span_tets, eout_prev,
                            rebuild ? -1 : first_new, surface_only);

            if (qhere && r == q_at) {   // this round: the -q nodes instead of the repairs (found again next round)
                const clk::time_point ta = clk::now();
                first_new = static_cast<int>(nd.size());
                st.q_added = quality_refine(g, m, nd, q);
                rebuild = false;

                if (std::getenv("V2M_TESS_TIMING")) {
                    std::fprintf(stderr, "[tt] q-refine   %8.0f ms (%zu nodes)\n", since(ta), st.q_added);
                }

                if (st.q_added > 0) {
                    continue;
                }
            }

            if ((fixes.empty() && ffix.empty()) || r >= budget) {
                break;
            }

            const clk::time_point ta = clk::now();
            size_t moved = 0;
            first_new = static_cast<int>(nd.size());
            // moves batched: a moved node costs a full Delaunay rebuild (~4.5 s at
            // 2.7M nodes) where an insertion is incremental (~0.15 s), and a round
            // moved only a few nodes (2-50) -- so each kind of repair first tries
            // insertions only, its moves (junction promotions) only once those are
            // dry (crossings before faces, as before) or in the last round
            // (with V2M_BATCH_MOVES=1; by default moves every round)
            const bool last = r + 1 >= budget;
            const bool pr0 = (promote || promote_first) && allow_promote && r == 0;
            size_t n = 0;

            for (int kind = 0; kind < 2 && n == 0; ++kind) {   // crossings / junctions, then faces
                const std::vector<Fix>& fx = kind == 0 ? fixes : ffix;

                if (fx.empty()) {
                    continue;
                }

                n = apply_fixes(g, fx, nd, &moved, kind == 0 && pr0, ngrid, eager_moves || last);

                if (n == 0 && !(eager_moves || last)) {
                    n = apply_fixes(g, fx, nd, &moved, false, ngrid, true);
                }
            }

            rebuild = moved > 0;   // moved nodes: no deletion in the live Delaunay

            if (std::getenv("V2M_TESS_TIMING")) {
                std::fprintf(stderr, "[tt] apply      %8.0f ms (%zu fixes, %zu moved)\n", since(ta), n, moved);
            }

            st.repaired += n;
            ++st.repair_rounds;

            if (n == 0) {
                break;
            }
        }
    };
    repair_loop(true);
    phase("repair");

    // radius-edge refinement (-q): circumcentres of the bad tets (onto the interface
    // where they encroach), inserted into the live Delaunay; each round is followed
    // by the conformity repairs again, so the new nodes cannot break conformity

    // one round by default (V2M_Q_ROUNDS): on Colin27 a second round was rolled
    // back, and a rollback is a full Delaunay rebuild (~7 s)
    static const int qrounds = std::getenv("V2M_Q_ROUNDS") ? std::atoi(std::getenv("V2M_Q_ROUNDS")) : 1;

    for (int r = 0; q > 0.0 && !q_inline && r < qrounds; ++r) {
        // transactional: a round whose repairs leave more non-conforming faces /
        // spanning tets / exterior edges than before is rolled back (nodes restored,
        // Delaunay rebuilt) and the refinement stops -- quality never costs conformity
        const size_t before = st.bad_faces + st.bad_span + st.bad_edges;
        const Nodes saved = nd;
        first_new = static_cast<int>(nd.size());
        const size_t na = quality_refine(g, m, nd, q);
        phase("q-refine");

        if (na == 0) {
            break;
        }

        rebuild = false;
        repair_loop(false);
        phase("q-repair");

        if (st.bad_faces + st.bad_span + st.bad_edges > before) {
            nd = saved;
            ngrid.valid = false;
            rebuild = true;
            tessellate_once(g, nd, voxel_mode, m, st, live, true, ffix, fixes, span_tets, eout_prev, -1, surface_only);
            st.q_rolled_back = 1;
            break;
        }

        st.q_added += na;
    }

    if (ss && ss->iters > 0) {   // the region surfaces smoothed (volume-preserving), before the interior's ODT
        const clk::time_point ts0 = clk::now();
        // (--mode surface: its tets are only the surfaces' scaffold -- flat, never
        // written: the triangles alone are guarded)
        if (surface_only) {
            m.P_orient = nd.P;
        }

        st.surf_moved = smooth_surfaces(nd, m, *ss, !surface_only, st.surf_nodes, st.surf_blocked);
        st.ms_surf = since(ts0);
        phase("surf-smooth");
    }

    if (smooth > 0) {
        // ODT smoothing alternated with re-Delaunay (Alliez et al. 2005): the moves
        // alone cannot open a sliver pinned by its neighbours, a new Delaunay can.
        // V2M_ODT_ROUNDS (default 1: a re-Delaunay round gained little and can break
        // the conformity of junction repairs) rounds; the last keeps the smoothed tets.
        static const int rounds = std::getenv("V2M_ODT_ROUNDS") ? std::atoi(std::getenv("V2M_ODT_ROUNDS")) : 1;
        const clk::time_point ts0 = clk::now();
        st.smoothed = 0;

        for (int r = 0; r < rounds; ++r) {
            st.smoothed += smooth_interior(g, nd, m, smooth);

            if (r + 1 < rounds) {
                tessellate_once(g, nd, voxel_mode, m, st, live, true, ffix, fixes, span_tets, eout_prev, -1, surface_only);
            }
        }

        st.ms_smooth = since(ts0);
        phase("smooth");
    }

    deviation_metrics(g, nd, ffix, span_tets, st);
    phase("deviation");

    if (opt) {   // sliver repair (ported from gpu_brain2mesh): flips, collapses, Steiner, smoothing
        OptParams op;
        op.verbose = std::getenv("V2M_OPT_VERBOSE") != nullptr;
        op.q = q;

        if (const char* e = std::getenv("V2M_OPT_ROUNDS")) {
            op.max_rounds = std::atoi(e);
        }
        OptStats os;
        optimize_mesh(m, nd, op, os);
        st.opt_flips32 = os.flips32;
        st.opt_flips23 = os.flips23;
        st.opt_collapses = os.collapses;
        st.opt_steiner = os.steiner;
        st.opt_moves = os.moves;
        st.opt_kites = os.kites;
        st.ms_opt = os.ms;

        if (std::getenv("V2M_TESS_TIMING")) {
            std::fprintf(stderr, "[tt] opt: 3-2 %.0f, kites %.0f, 2-3 %.0f, collapse %.0f, Steiner %.0f, smooth %.0f ms (%d rounds)\n",
                         os.ms_pass[0], os.ms_pass[1], os.ms_pass[2], os.ms_pass[3], os.ms_pass[4], os.ms_pass[5], os.rounds);
        }
    }

    phase("opt");
    mesh_quality(g, nd, m, st);
    phase("quality");
}

// (declared in v2m_gdel.h; here, so that the CPU-only build has it too)
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
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

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

}  // namespace tn
