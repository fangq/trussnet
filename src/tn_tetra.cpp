// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_tetra.cpp -- see tn_tetra.h.

#include "tn_tetra.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <unordered_map>

#include "delaunay.h"
#include "tn_omp.h"

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
#define TN_G
#include "opencl/tn_grid_body.cl"
#include "opencl/tn_seed_body.cl"
#include "opencl/tn_particle_body.cl"
#undef TN_G

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

}  // namespace

// A repair: the crossing of edge (x, y) with the a|b interface (b = 0: the
// exterior).
struct Fix {
    uint32_t x, y;   // y == UINT32_MAX: a junction fix at the centroid of tet x's nodes;
                     // y == TN_FACEFIX: a face fix, the centroid of face tv[0..2] onto a|b
    int a, b;
    uint32_t tv[4];
};

static const uint32_t TN_FACEFIX = UINT32_MAX - 1;

// Unique edges (x < y) of a tet list, in parallel: node -> tet CSR (count ->
// scan -> fill), then each node x collects its neighbours y > x from its tets and
// sorts / uniques that short list. Replaces a serial sort of 6 pairs per tet.
static std::vector<std::pair<int, int>> unique_edges(const std::vector<int>& tets, int nn) {
    const int64_t nt = static_cast<int64_t>(tets.size() / 4);
    std::vector<int> cnt(static_cast<size_t>(nn) + 1, 0);

    for (int64_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            ++cnt[tets[4 * t + k] + 1];
        }

    for (int i = 0; i < nn; ++i) {
        cnt[i + 1] += cnt[i];
    }

    std::vector<int> cur(cnt.begin(), cnt.end() - 1), n2t(static_cast<size_t>(cnt[nn]));

    for (int64_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            n2t[cur[tets[4 * t + k]]++] = static_cast<int>(t);
        }

    std::vector<int> ecnt(static_cast<size_t>(nn) + 1, 0);
    std::vector<std::vector<int>> nb(nn);
    #pragma omp parallel for schedule(dynamic, 1024)

    for (int x = 0; x < nn; ++x) {
        std::vector<int>& L = nb[x];

        for (int s = cnt[x]; s < cnt[x + 1]; ++s)
            for (int k = 0; k < 4; ++k) {
                const int y = tets[4 * static_cast<int64_t>(n2t[s]) + k];

                if (y > x) {
                    L.push_back(y);
                }
            }

        std::sort(L.begin(), L.end());
        L.erase(std::unique(L.begin(), L.end()), L.end());
        ecnt[x + 1] = static_cast<int>(L.size());
    }

    for (int x = 0; x < nn; ++x) {
        ecnt[x + 1] += ecnt[x];
    }

    std::vector<std::pair<int, int>> e(static_cast<size_t>(ecnt[nn]));
    #pragma omp parallel for schedule(dynamic, 1024)

    for (int x = 0; x < nn; ++x)
        for (size_t k = 0; k < nb[x].size(); ++k) {
            e[ecnt[x] + k] = std::make_pair(x, nb[x][k]);
        }

    return e;
}

// label set of node v: {a}, {a,b} interface, {a,b,c} junction, {a,b,c,d} corner
static int node_label_set(const Nodes& nd, uint32_t v, int* out) {
    out[0] = nd.lab[v];
    int n = 1;

    if (nd.typ[v] != TN_INTERIOR) {
        out[n++] = nd.part[2 * v];
    }

    if (nd.typ[v] >= TN_JUNCTION) {
        out[n++] = nd.part[2 * v + 1];
    }

    if (nd.typ[v] == TN_CORNER) {
        out[n++] = nd.part3[v];
    }

    return n;
}

// `live` persists across repair rounds: nullptr or rebuild -> a fresh Delaunay of
// all nodes; else the nodes appended since the last round are inserted into it
// incrementally (the repairs only ADD nodes then; a round that moved nodes
// asks for a rebuild).
static void tessellate_once(const Grid& g, const Nodes& nd, bool voxel_mode, TetOut& m, TetStats& st,
                            std::unique_ptr<::TetMesh>& live, bool rebuild, std::vector<Fix>& facefixes,
                            std::vector<Fix>& fixes) {
    OmpThreadCap cap;
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
    const int n = static_cast<int>(nd.size());
    std::vector<double> X(nd.P.begin(), nd.P.end());

    const bool timing = std::getenv("TN_TESS_TIMING") != nullptr;
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
        live->init_vertices(X.data(), static_cast<uint32_t>(n));
        live->tetrahedrize();
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
    // (TN_CIRCUMCENTRE_LABEL: the plain restricted-Delaunay rule, for comparison;
    // on the relaxed nodes it dropped 1% of a convex ball's volume -- flat surface
    // tets whose circumcentre falls just outside.)
    const bool use_cc = std::getenv("TN_CIRCUMCENTRE_LABEL") != nullptr;
    clk::time_point t1 = clk::now();
    std::vector<int> tl(static_cast<size_t>(nt), -1);   // -1 ghost
    std::vector<char> conflict(static_cast<size_t>(nt), 0);
    auto node_labels = [&](uint32_t v, int* out) {
        return node_label_set(nd, v, out);
    };
    #pragma omp parallel for schedule(dynamic, 4096)

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
        const int lc = tn_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(),
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
        } else {   // several (a flat tet on one interface / junction): the centroid decides
            int best = I[0];
            float pb = -1.0f;

            for (int x = 0; x < ni; ++x) {
                if (I[x] == lc) {
                    best = lc;
                    break;
                }

                const float w = tn_phi_at(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                          g.phi.data(), I[x], static_cast<float>(o[0]), static_cast<float>(o[1]),
                                          static_cast<float>(o[2]));

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
    // depth of a sample below the exterior surface, from the VOXELS (exact and
    // independent of the interface field; |psi| / |grad psi| blows up in the sharp
    // thin-layer field): 0 in a non-exterior voxel, else the distance to the
    // nearest face between an exterior voxel and a voxel of `sec` (the runner-up)
    auto outside_depth = [&](float x, float y, float z, int sec) {
        const int vi = static_cast<int>(std::floor(x / g.vs[0] + 0.5f)), vj = static_cast<int>(std::floor(y / g.vs[1] + 0.5f)),
                  vk = static_cast<int>(std::floor(z / g.vs[2] + 0.5f));

        if (tn_label_at(g.L->data(), g.nx, g.ny, g.nz, vi, vj, vk) != 0) {
            return 0.0f;
        }

        if (sec == TN_NOLAB || sec == 0) {
            return 1e30f;
        }

        const float q[3] = { x, y, z };
        float yv[3];
        const float d2 = tn_vox_nearest(d, g.L->data(), 0, sec, TN_NOLAB, q, 3, yv);
        return d2 < 0.0f ? 1e30f : std::sqrt(d2);
    };
    auto tn_h_at_voxel = [&](const float* p) {
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
                const int b = tn_brick_of(d, vi, vj, vk);
                bool has0 = false;

                for (int e = 0; e < g.bl_cnt[b] && e < TN_BL; ++e) {
                    has0 |= g.bl_lab[static_cast<size_t>(b) * TN_BL + e] == 0;
                }

                if (!has0) {
                    continue;
                }
            }
            int sec;
            float mg;
            const int l = tn_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(),
                                      voxel_mode ? 1 : 0, p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]),
                                      p[2] + tt * (q[2] - p[2]), &sec, &mg);

            if (l == 0 && outside_depth(p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1]), p[2] + tt * (q[2] - p[2]),
                                        sec) > 0.2f * tn_h_at_voxel(p) + 0.87f * vmin0) {   // + the staircase half-diagonal
                return true;
            }
        }

        return false;
    };
    auto peelable = [&](int64_t t) {
        const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);

        for (int k = 0; k < 4; ++k)
            if (!on_exterior(v[k])) {
                return false;
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

            if (tn_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(),
                            voxel_mode ? 1 : 0, static_cast<float>(o[0]), static_cast<float>(o[1]),
                            static_cast<float>(o[2]), &sec, &mg) == 0) {
                return true;
            }
        }

        return false;
    };
    st.peeled = 0;

    for (int round = 0; round < 64; ++round) {
        std::vector<int64_t> cand;

        for (int64_t t = 0; t < nt; ++t) {
            if (tl[t] <= 0) {
                continue;
            }

            const uint64_t* nb = tin.getTetNeighs(static_cast<uint64_t>(t) * 4);
            bool bnd = false;

            for (int f = 0; f < 4 && !bnd; ++f) {
                bnd = tl[nb[f] >> 2] <= 0;
            }

            if (bnd) {
                cand.push_back(t);
            }
        }

        std::vector<char> rm(cand.size(), 0);
        #pragma omp parallel for schedule(dynamic, 256)

        for (int64_t c = 0; c < static_cast<int64_t>(cand.size()); ++c) {
            rm[c] = peelable(cand[c]);
        }

        size_t np = 0;

        for (size_t c = 0; c < cand.size(); ++c)
            if (rm[c]) {
                tl[cand[c]] = 0;
                ++np;
            }

        st.peeled += np;

        if (np == 0) {
            break;
        }
    }

    lap("sculpt");

    // 3. kept tets and the conformity checks
    clk::time_point t2 = clk::now();
    m.P = nd.P;
    m.tets.clear();
    m.label.clear();
    st.delaunay_tets = 0;

    for (int64_t t = 0; t < nt; ++t) {
        if (tl[t] < 0) {
            continue;
        }

        ++st.delaunay_tets;

        if (tl[t] == 0) {
            continue;
        }

        const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);

        for (int k = 0; k < 4; ++k) {
            m.tets.push_back(static_cast<int32_t>(v[k]));
        }

        m.label.push_back(tl[t]);
    }

    st.kept = m.label.size();
    // node is on the interface between labels x and y?
    auto on_iface = [&](int i, int x, int y) {
        if (nd.typ[i] == TN_INTERIOR) {
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
    FILE* dbg = std::getenv("TN_TESS_DEBUG") ? std::fopen(std::getenv("TN_TESS_DEBUG"), "wb") : nullptr;

    for (int64_t t = 0; t < nt; ++t) {
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
                    ninter += nd.typ[v[k]] == TN_INTERIOR;

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

                if (std::getenv("TN_TESS_VERBOSE")) {   // the nodes of every failing tet
                    int sc;
                    float mg;
                    const int lc2 = tn_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                                g.phi.data(), 0, r[0], r[1], r[2], &sc, &mg);
                    std::fprintf(stderr, "[fail] tet %lld %s label %d, centroid (%.2f %.2f %.2f) field %d\n",
                                 static_cast<long long>(t), conflict[t] ? "SPAN" : "FACE", tl[t], r[0], r[1], r[2], lc2);

                    for (int k = 0; k < 4; ++k) {
                        int S3[4];
                        const int n3 = node_labels(v[k], S3);
                        const float* q = &nd.P[3 * v[k]];
                        const int lq = tn_label_of(d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(),
                                                   g.phi.data(), 0, q[0], q[1], q[2], &sc, &mg);
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
                    fx.y = TN_FACEFIX;
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
    std::vector<char> eout(edges.size(), 0);   // cached for the repairs below
    size_t bad_edges = 0;
    #pragma omp parallel for schedule(dynamic, 4096) reduction(+ : bad_edges)

    for (int64_t e = 0; e < static_cast<int64_t>(edges.size()); ++e) {
        eout[e] = edge_outside(static_cast<uint32_t>(edges[e].first), static_cast<uint32_t>(edges[e].second)) ? 1 : 0;
        bad_edges += eout[e];
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

    // deviation of the offending nodes (voxels)
    {
        const float vm = std::min(g.vs[0], std::min(g.vs[1], g.vs[2]));
        // exact, field-independent: the distance to the nearest face between an l1
        // voxel and an l2 voxel (either orientation), searched within 4 voxels
        auto dist = [&](int l1, int l2, const float* p) {
            float y[3];
            float d2 = tn_vox_nearest(d, g.L->data(), l1, l2, TN_NOLAB, p, 4, y);
            const float e2 = tn_vox_nearest(d, g.L->data(), l2, l1, TN_NOLAB, p, 4, y);

            if (d2 < 0.0f || (e2 >= 0.0f && e2 < d2)) {
                d2 = e2;
            }

            return d2 < 0.0f ? 5.0f : std::sqrt(d2) / vm;
        };
        auto pct = [](std::vector<float>& x, double* out) {
            if (x.empty()) {
                return;
            }

            std::sort(x.begin(), x.end());
            out[0] = x[x.size() / 2];
            out[1] = x[static_cast<size_t>(0.95 * (x.size() - 1))];
            out[2] = x[static_cast<size_t>(0.99 * (x.size() - 1))];
            out[3] = x.back();
        };
        std::vector<float> df, ds;

        for (const Fix& f : facefixes) {   // face nodes not on the a|b interface (this round)
            float w = 0.0f;

            for (int k = 0; k < 3; ++k)
                if (!on_iface(static_cast<int>(f.tv[k]), f.a, f.b)) {
                    w = std::max(w, dist(f.a, f.b, &nd.P[3 * f.tv[k]]));
                }

            df.push_back(w);
        }

        for (int64_t t = 0; t < nt; ++t) {   // spanning: nodes lacking the tet's label
            if (tl[t] <= 0 || !conflict[t]) {
                continue;
            }

            const uint32_t* v = tin.getTetNodes(static_cast<uint64_t>(t) * 4);
            float w = 0.0f;

            for (int k = 0; k < 4; ++k) {
                int S[4];
                const int n = node_labels(v[k], S);
                bool has = false;
                float dm = 99.0f;

                for (int e = 0; e < n; ++e) {
                    has |= S[e] == tl[t];
                }

                if (has) {
                    continue;
                }

                for (int e = 0; e < n; ++e) {
                    dm = std::min(dm, dist(S[e], tl[t], &nd.P[3 * v[k]]));
                }

                w = std::max(w, dm);
            }

            ds.push_back(w);
        }

        for (int k = 0; k < 4; ++k) {
            st.dev_face[k] = st.dev_span[k] = 0.0;
        }

        pct(df, st.dev_face);
        pct(ds, st.dev_span);
    }

    st.ms_check = since(t2);
    lap("deviation");

}


// Apply the repairs: the crossing point of each edge with its interface (the
// exterior surface for b = 0) is projected onto it; if an endpoint is within
// 0.3 h it is PROMOTED onto the interface (moved there), else a new interface
// (or junction) node is added. New points closer than 0.3 h to one already added
// this round are skipped.
static size_t apply_fixes(const Grid& g, const std::vector<Fix>& fixes, Nodes& nd, size_t* moved, bool promote) {
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
#define FLD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data()
    std::vector<char> touched(nd.size(), 0);
    size_t nfix = 0, sk_same = 0, sk_out = 0, sk_sign = 0, sk_near = 0;
    // every node in a hash grid of cell 0.3 hmin: a new or promoted position must
    // keep >= 0.3 h from all other nodes (coincident points break the exact
    // Delaunay's symbolic perturbation)
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

    for (const Fix& f : fixes) {
        if (f.y == TN_FACEFIX) {   // the face centroid, projected onto the a|b interface
            const int a = f.a == 0 ? f.b : f.a, b = f.a == 0 ? 0 : f.b;
            float c[3] = { 0, 0, 0 };

            for (int k = 0; k < 3; ++k)
                for (int e = 0; e < 3; ++e) {
                    c[e] += nd.P[3 * f.tv[k] + e] / 3.0f;
                }

            const float h = tn_h_at(d, g.h.data(), c[0], c[1], c[2]);

            if (a == b || !tn_project1(FLD, a, b, c, 0.5f * h) || !tn_valid_on(FLD, a, b, TN_NOLAB, c) ||
                    near_node(c, 0.3f * h, -1, -1) >= 0) {
                ++sk_ff;
                continue;
            }

            int typ = TN_INTERFACE, cc = TN_NOLAB;
            const int third = tn_third_label(FLD, a, b, c);

            if (third != TN_NOLAB) {
                float r[3] = { c[0], c[1], c[2] };

                if (tn_project2(FLD, a, b, third, r, 0.5f * h) && tn_valid_on(FLD, a, b, third, r) &&
                        near_node(r, 0.3f * h, -1, -1) < 0) {
                    c[0] = r[0];
                    c[1] = r[1];
                    c[2] = r[2];
                    typ = TN_JUNCTION;
                    cc = third;
                }
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(a));
            nd.typ.push_back(static_cast<uint8_t>(typ));
            nd.part.push_back(static_cast<uint16_t>(b));
            nd.part.push_back(static_cast<uint16_t>(cc));
            nd.part3.push_back(TN_NOLAB);
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

            const float h = tn_h_at(d, g.h.data(), c[0], c[1], c[2]);

            if (!tn_project2(FLD, ja, jb, jc, c, 0.5f * h) || !tn_valid_on(FLD, ja, jb, jc, c) || near_node(c, 0.3f * h, -1, -1) >= 0) {
                ++sk_jf;
                continue;
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(ja));
            nd.typ.push_back(TN_JUNCTION);
            nd.part.push_back(static_cast<uint16_t>(jb));
            nd.part.push_back(static_cast<uint16_t>(jc));
            nd.part3.push_back(TN_NOLAB);
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
            const int l = tn_label_of(FLD, 0, r[0], r[1], r[2], &sec, &mg);
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

        if (!(tn_psi(FLD, a2, b2, lo[0], lo[1], lo[2]) > 0.0f && tn_psi(FLD, a2, b2, hi[0], hi[1], hi[2]) <= 0.0f)) {
            ++sk_sign;
            continue;
        }

        for (int k = 0; k < 3; ++k) {
            p[k] = lo[k];
            q[k] = hi[k];
        }

        const int a = a2 == 0 ? b2 : a2, b = a2 == 0 ? 0 : b2;   // own label never 0
        tn_crossing(FLD, a2, b2, p, q, c);
        const float h = tn_h_at(d, g.h.data(), c[0], c[1], c[2]);
        tn_project1(FLD, a, b, c, 0.25f * h);   // polish; the crossing is already on it
        int typ = TN_INTERFACE, cc = TN_NOLAB;
        const int third = tn_third_label(FLD, a, b, c);

        if (third != TN_NOLAB) {
            float r[3] = { c[0], c[1], c[2] };
            if (tn_project2(FLD, a, b, third, r, 0.5f * h)) {
                c[0] = r[0];
                c[1] = r[1];
                c[2] = r[2];
                typ = TN_JUNCTION;
                cc = third;
            }
        }

        // promote an endpoint that is already close
        const uint32_t ends[2] = { f.x, f.y };
        bool done = false;

        // (first repair round only: a moved node forces a full Delaunay rebuild,
        // while insert-only rounds are incremental)

        for (int e = 0; e < 2 && !done && promote; ++e) {
            const uint32_t v = ends[e];
            const float dx = nd.P[3 * v] - c[0], dy = nd.P[3 * v + 1] - c[1], dz = nd.P[3 * v + 2] - c[2];

            if (!touched[v] && nd.typ[v] == TN_INTERIOR && dx * dx + dy * dy + dz * dz < 0.09f * h * h &&
                    (nd.lab[v] == a || nd.lab[v] == b) && near_node(c, 0.3f * h, v, v) < 0) {
                ++*moved;
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
            if (near_node(c, 0.3f * h, -1, -1) >= 0) {
                ++sk_near;
                continue;
            }

            grid[ckey(c)].push_back(static_cast<uint32_t>(nd.size()));
            nd.P.insert(nd.P.end(), c, c + 3);
            nd.lab.push_back(static_cast<uint16_t>(a == 0 ? b : a));
            nd.typ.push_back(static_cast<uint8_t>(typ));
            nd.part.push_back(static_cast<uint16_t>(a == 0 ? 0 : b));
            nd.part.push_back(static_cast<uint16_t>(cc));
            nd.part3.push_back(TN_NOLAB);
            touched.push_back(1);
        }

        ++nfix;
    }

#undef FLD

    if (std::getenv("TN_REPAIR_DEBUG")) {
        std::fprintf(stderr, "[repair] %zu fixes: %zu applied (%zu junction, %zu face); skipped: %zu same-label, %zu no "
                     "outside sample, %zu no sign change, %zu near a node, %zu junction, %zu face\n", fixes.size(), nfix,
                     njf, nff, sk_same, sk_out, sk_sign, sk_near, sk_jf, sk_ff);
    }

    return nfix;
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

    std::vector<double> jl(m.label.size());
    double mind = 180.0, vol = 0.0;
    size_t s10 = 0, s5 = 0;

    for (size_t t = 0; t < m.label.size(); ++t) {
        double q[4][3];
        const double* pp[4];

        for (int k = 0; k < 4; ++k) {
            for (int c = 0; c < 3; ++c) {
                q[k][c] = nd.P[3 * m.tets[4 * t + k] + c];
            }

            pp[k] = q[k];
        }

        double md, j, v;
        tet_quality(pp, md, j, v);
        jl[t] = j;

        if (m.label[t] >= static_cast<int>(st.label_vol.size())) {
            st.label_vol.resize(m.label[t] + 1, 0.0);
        }

        st.label_vol[m.label[t]] += std::fabs(v);
        mind = std::min(mind, md);
        s10 += md < 10.0;
        s5 += md < 5.0;
        vol += v;
    }

    st.min_dihedral = mind;
    st.slivers10 = s10;
    st.slivers5 = s5;
    st.volume = vol;

    if (!jl.empty()) {
        std::vector<double> s(jl);
        std::sort(s.begin(), s.end());
        st.joe_liu_min = s.front();
        st.joe_liu_p5 = s[s.size() / 20];
        st.joe_liu_med = s[s.size() / 2];
    }

}

void tessellate(const Grid& g, Nodes& nd, bool voxel_mode, int max_repair, TetOut& m, TetStats& st) {
    std::vector<Fix> fixes, ffix;
    std::unique_ptr<::TetMesh> live;
    bool rebuild = true;
    st.repair_rounds = 0;
    st.repaired = 0;

    for (int r = 0;; ++r) {
        tessellate_once(g, nd, voxel_mode, m, st, live, rebuild, ffix, fixes);

        if ((fixes.empty() && ffix.empty()) || r >= max_repair) {
            break;
        }

        const clk::time_point ta = clk::now();
        size_t moved = 0;
        size_t n = fixes.empty() ? 0 : apply_fixes(g, fixes, nd, &moved, r == 0);

        if (n == 0 && !ffix.empty()) {   // crossing / junction repairs exhausted: faces
            n = apply_fixes(g, ffix, nd, &moved, false);
        }

        rebuild = moved > 0;   // moved nodes: no deletion in the live Delaunay

        if (std::getenv("TN_TESS_TIMING")) {
            std::fprintf(stderr, "[tt] apply      %8.0f ms (%zu fixes, %zu moved)\n", since(ta), n, moved);
        }

        st.repaired += n;
        ++st.repair_rounds;

        if (n == 0) {
            break;
        }
    }

    mesh_quality(g, nd, m, st);
}

}  // namespace tn
