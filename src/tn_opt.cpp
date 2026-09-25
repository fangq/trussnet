// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (the optimisation passes are ported from gpu_brain2mesh src/opencl/
//  b2m_refine_cl.cpp, same author, GPL-3.0-or-later)
//
// tn_opt.cpp -- see tn_opt.h. Sliver repair on the final labelled mesh: 3-2 and
// 2-3 flips, interior edge collapse, sliver Steiner points and guarded interior
// smoothing, iterated while anything changes. Every operation is monotone in the
// worst minimum dihedral angle of the patch it touches and never crosses a
// constrained face (a face between two labels, or on the exterior), so region
// labels and interfaces are preserved; trussnet's interface / junction / corner
// nodes are additionally frozen.

#include "tn_opt.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "tn_log.h"
#include "tn_omp.h"
#include "tn_particles.h"

#ifndef TN_INTERIOR
    #define TN_INTERIOR 0   // node types / "no label" as in src/opencl/tn_seed_body.cl
#endif
#ifndef TN_NOLAB
    #define TN_NOLAB 0xFFFF
#endif

namespace tn {

namespace {

// the mesh the ported passes operate on (the subset of gpu_brain2mesh's CoarseCDT
// they use): face i of a tet is opposite local vertex i
struct CoarseCDT {
    std::vector<double> points;                 // 3 per point
    std::vector<int> tets;                      // 4 per tet, positively oriented
    std::vector<int> tet_label;
    std::vector<int> tet_neigh;                 // 4 per tet, -1 = none
    std::vector<unsigned char> tet_face_marker; // 4 per tet: 1 = constrained face
    std::vector<int> point_tet;
    std::vector<unsigned char> point_marker;    // 1 = frozen node
    std::vector<int> point_orig;                // trussnet node index, -1 = new
    std::vector<unsigned char> point_failed;    // smoothing: last move attempt failed ...
    std::vector<uint64_t> point_sig;            // ... on the patch with this signature
    int64_t numPoints() const {
        return static_cast<int64_t>(points.size() / 3);
    }
    int64_t numTets() const {
        return static_cast<int64_t>(tets.size() / 4);
    }
};

// no refinement criteria here (the guard of the gpu_brain2mesh optimiser keeps
// its refinement guarantees; trussnet sizes the mesh upstream)
static bool g_opt_guard = false;   // set per run: a collapse may not create a tet with radius-edge > q
static double g_opt_q = 0.0;
static const std::array<double, 6> g_opt_sz = { { 0, 0, 0, 0, 0, 0 } };
static const std::array<double, 6> g_opt_vol = { { 0, 0, 0, 0, 0, 0 } };
// radius-edge ratio test of b2m_check_bad (the size / volume / dihedral caps of
// gpu_brain2mesh's refiner are not used here): 1 if circumradius / shortest edge
// exceeds minratio
static inline int b2m_check_bad(const double* pa, const double* pb, const double* pc, const double* pd, double minratio,
                                double, double, double) {
    if (minratio <= 0.0) {
        return 0;
    }

    const double u[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] }, v[3] = { pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2] },
                 w[3] = { pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2] };
    const double vxw[3] = { v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0] };
    const double wxu[3] = { w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0] };
    const double uxv[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
    const double det = 2.0 * (u[0] * vxw[0] + u[1] * vxw[1] + u[2] * vxw[2]);

    if (std::fabs(det) < 1e-300) {
        return 1;
    }

    const double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2], vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
                 ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    double r2 = 0.0;

    for (int k = 0; k < 3; ++k) {
        const double o = (uu * vxw[k] + vv * wxu[k] + ww * uxv[k]) / det;
        r2 += o * o;
    }

    const double e2[6] = { uu, vv, ww, (pc[0] - pb[0]) * (pc[0] - pb[0]) + (pc[1] - pb[1]) * (pc[1] - pb[1]) + (pc[2] - pb[2]) * (pc[2] - pb[2]),
                           (pd[0] - pb[0]) * (pd[0] - pb[0]) + (pd[1] - pb[1]) * (pd[1] - pb[1]) + (pd[2] - pb[2]) * (pd[2] - pb[2]),
                           (pd[0] - pc[0]) * (pd[0] - pc[0]) + (pd[1] - pc[1]) * (pd[1] - pc[1]) + (pd[2] - pc[2]) * (pd[2] - pc[2]) };
    const double emin = *std::min_element(e2, e2 + 6);
    return r2 > minratio * minratio * emin ? 1 : 0;
}

// non-robust orient3d (positive if d is below the oriented plane abc)
static inline double b2m_orient3d(const double* a, const double* b, const double* c, const double* d) {
    const double adx = a[0] - d[0], ady = a[1] - d[1], adz = a[2] - d[2];
    const double bdx = b[0] - d[0], bdy = b[1] - d[1], bdz = b[2] - d[2];
    const double cdx = c[0] - d[0], cdy = c[1] - d[1], cdz = c[2] - d[2];
    return adx * (bdy * cdz - bdz * cdy) - ady * (bdx * cdz - bdz * cdx) + adz * (bdx * cdy - bdy * cdx);
}


static inline std::array<int, 3> face_key(int a, int b, int c) {
    std::array<int, 3> k = { { a, b, c } };
    std::sort(k.begin(), k.end());
    return k;
}

struct FaceKeyHash {
    size_t operator()(const std::array<int, 3>& k) const {
        uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(k[0])) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(k[1])) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(k[2])) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        return static_cast<size_t>(h);
    }
};

template <class T>
static void par_sort(std::vector<T>& v) {
    const size_t n = v.size();
#ifdef _OPENMP
    const int nth = omp_get_max_threads();
#else
    const int nth = 1;
#endif

    if (nth <= 1 || n < 65536) {
        std::sort(v.begin(), v.end());
        return;
    }

    int nc = 1;

    while (nc < nth) {
        nc <<= 1;    // power-of-two chunk count for the pairwise merge tree
    }

    std::vector<size_t> bnd(static_cast<size_t>(nc) + 1);

    for (int c = 0; c <= nc; ++c) {
        bnd[c] = n * static_cast<size_t>(c) / static_cast<size_t>(nc);
    }

    #pragma omp parallel for schedule(static)

    for (int c = 0; c < nc; ++c) {
        std::sort(v.begin() + bnd[c], v.begin() + bnd[c + 1]);
    }

    for (int w = 1; w < nc; w <<= 1) {
        #pragma omp parallel for schedule(static)

        for (int c = 0; c < nc; c += 2 * w) {
            std::inplace_merge(v.begin() + bnd[c], v.begin() + bnd[c + w], v.begin() + bnd[c + 2 * w]);
        }
    }
}

// Compact away dead tets, then REBUILD tet_neigh from scratch by global face-hash.
// The sequential insertion maintains adjacency well enough for its own BFS, but the
// hand surgery leaves a few stale pointers (a tet whose neighbour was later killed by
// another insertion); rebuilding from the face incidence is consistent by
// construction (defects=0), the same method build_coarse_cdt uses.
static void compact_dead_cpu(CoarseCDT& m, const std::vector<char>& dead) {
    const int64_t nt = m.numTets();
    int k = 0;

    for (int64_t t = 0; t < nt; ++t) {
        if (dead[t]) {
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            m.tets[4 * k + i]            = m.tets[4 * t + i];
            m.tet_face_marker[4 * k + i] = m.tet_face_marker[4 * t + i];
        }

        m.tet_label[k] = m.tet_label[t];
        ++k;
    }

    m.tets.resize(static_cast<size_t>(k) * 4);
    m.tet_face_marker.resize(static_cast<size_t>(k) * 4);
    m.tet_label.resize(static_cast<size_t>(k));
    // Rebuild adjacency by SORTING all faces (parallel collect + par_sort) and pairing
    // equal neighbours -- a hash map over every face was the serial bottleneck of the
    // flip/collapse passes on large meshes. A face met more than twice (invalid)
    // pairs its first two occurrences, as before.
    m.tet_neigh.assign(static_cast<size_t>(k) * 4, -1);
    {
        // trussnet: faces bucketed by their smallest vertex (counting sort, linear)
        // and matched inside each small bucket in parallel -- same pairing as the
        // global par_sort of all faces it replaces (~0.7 s per call on 5M tets)
        const int64_t nf = static_cast<int64_t>(k) * 4;
        const int64_t npnt = m.numPoints();
        std::vector<int> bcnt(static_cast<size_t>(npnt) + 1, 0), bfill;
        std::vector<std::array<int, 3>> fkey(static_cast<size_t>(nf));
        #pragma omp parallel for schedule(static)

        for (int64_t t = 0; t < k; ++t)
            for (int f = 0; f < 4; ++f) {
                int v[3], c = 0;

                for (int j2 = 0; j2 < 4; ++j2) if (j2 != f) {
                        v[c++] = m.tets[4 * t + j2];
                    }

                fkey[4 * t + f] = face_key(v[0], v[1], v[2]);
            }

        for (int64_t e = 0; e < nf; ++e) {
            ++bcnt[fkey[e][0] + 1];
        }

        for (int64_t v = 0; v < npnt; ++v) {
            bcnt[v + 1] += bcnt[v];
        }

        bfill.assign(bcnt.begin(), bcnt.end() - 1);
        std::vector<int> order(static_cast<size_t>(nf));

        for (int64_t e = 0; e < nf; ++e) {
            order[bfill[fkey[e][0]]++] = static_cast<int>(e);
        }

        #pragma omp parallel for schedule(dynamic, 4096)

        for (int64_t v = 0; v < npnt; ++v) {
            int* b0 = order.data() + bcnt[v];
            int* b1 = order.data() + bcnt[v + 1];

            if (b1 - b0 < 2) {
                continue;
            }

            std::sort(b0, b1, [&](int x, int y) {
                return fkey[x][1] < fkey[y][1] || (fkey[x][1] == fkey[y][1] && (fkey[x][2] < fkey[y][2] ||
                                                   (fkey[x][2] == fkey[y][2] && x < y)));
            });

            for (int* it = b0; it + 1 < b1;) {
                if (fkey[it[0]] == fkey[it[1]]) {
                    const int a = it[0], b = it[1];
                    m.tet_neigh[a] = b >> 2;
                    m.tet_neigh[b] = a >> 2;
                    int* jt = it + 2;

                    while (jt < b1 && fkey[*jt] == fkey[*it]) {
                        ++jt;    // skip extra (non-manifold) occurrences
                    }

                    it = jt;
                } else {
                    ++it;
                }
            }
        }
    }

    // The inserted apex points grew m.points; resync the per-point arrays the rest of
    // the pipeline expects (insert_round_cpu reads point_marker up to numPoints, and
    // rebuilds point_tet). New circumcentre points are interior (marker 0).
    m.point_marker.resize(static_cast<size_t>(m.numPoints()), 0);
    m.point_tet.assign(static_cast<size_t>(m.numPoints()), -1);

    for (int64_t t = 0; t < m.numTets(); ++t)
        for (int i = 0; i < 4; ++i) {
            int v = m.tets[4 * t + i];

            if (m.point_tet[v] < 0) {
                m.point_tet[v] = static_cast<int>(t);
            }
        }
}

// ---------------------------------------------------------------------------
// Sliver removal by 3-2 flips (Klingner-Shewchuk style, vertex-preserving).
//
// The size-driven refiner cannot remove SLIVERS: a near-degenerate tet has a
// huge circumradius, so its Steiner point lands outside the domain and the bbox
// guard (correctly) rejects it. Smoothing is worse than useless on these thin-
// shell brain meshes (~89% of vertices sit on a constrained surface/interface,
// so a Laplacian pass drags them along the thin layers and MULTIPLIES slivers).
// The standard vertex-preserving cure is topological: where a sliver shares an
// interior edge with exactly 3 same-tissue tets, replace those 3 tets with 2
// (a 3-2 flip), which deletes the bad tet without moving any vertex or touching
// a constrained face. Empirically ~63% of the residual slivers are 3-2-flippable.
// We only apply a flip that strictly raises the worst min-dihedral of the patch,
// so the pass is monotone (it terminates) and never degrades quality.

static inline void b2m_cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

// Minimum dihedral angle (degrees) of the tet (pa,pb,pc,pd); 0 for a flat sliver.
static double b2m_tet_min_dihedral(const double* pa, const double* pb,
                                   const double* pc, const double* pd) {
    const double* P[4] = { pa, pb, pc, pd };
    static const int E[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}, {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
    // n1, n2 are both perpendicular to the edge, so the angle between them IS the
    // dihedral angle at that edge (a regular-corner tet gives 90/90/90/54.7). acos is
    // monotonic decreasing, so the minimum dihedral is acos(max cosine): one acos
    // instead of six (this function dominates every optimisation pass).
    double dmax = -1.0;

    for (int e = 0; e < 6; ++e) {
        double ev[3], a1[3], a2[3], n1[3], n2[3];

        for (int k = 0; k < 3; ++k) {
            ev[k] = P[E[e][1]][k] - P[E[e][0]][k];
            a1[k] = P[E[e][2]][k] - P[E[e][0]][k];
            a2[k] = P[E[e][3]][k] - P[E[e][0]][k];
        }

        b2m_cross3(ev, a1, n1);
        b2m_cross3(ev, a2, n2);
        double l1 = std::sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
        double l2 = std::sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
        double d = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2 + 1e-300);

        if (d > dmax) {
            dmax = d;
        }
    }

    dmax = dmax < -1.0 ? -1.0 : (dmax > 1.0 ? 1.0 : dmax);
    return std::acos(dmax) * 57.29577951308232;
}

// Dihedral-angle quality report: the metric that actually matters for a tet mesh
// (radius-edge, the refiner's bad-marker, does NOT bound the min angle -- a flat
// sliver can have a good radius-edge yet a ~0-degree dihedral). Prints a min-
// dihedral histogram + the count of inverted/degenerate tets so the effect of the
// optimization phase is measurable. `tag` labels the stage (e.g. "pre-opt").
static void quality_report(const CoarseCDT& m, const char* tag) {
    const double* P = m.points.data();
    const int64_t nt = m.numTets();

    if (nt == 0) {
        TN_FPRINTF(stderr, "[quality] %-8s empty\n", tag);
        return;
    }

    // buckets: [0,5) [5,10) [10,20) [20,30) [30,40) [40,180]
    long long h[6] = { 0, 0, 0, 0, 0, 0 };
    long long ninv = 0;
    double mn = 180.0, sum = 0.0;

    for (int64_t t = 0; t < nt; ++t) {
        const int* tv = &m.tets[4 * t];
        const double* a = &P[3 * tv[0]];
        const double* b = &P[3 * tv[1]];
        const double* c = &P[3 * tv[2]];
        const double* d = &P[3 * tv[3]];
        double v6 = b2m_orient3d(a, b, c, d);

        if (v6 == 0.0) {
            ++ninv;
        }

        double q = b2m_tet_min_dihedral(a, b, c, d);
        sum += q;

        if (q < mn) {
            mn = q;
        }

        int bi = q < 5 ? 0 : q < 10 ? 1 : q < 20 ? 2 : q < 30 ? 3 : q < 40 ? 4 : 5;
        ++h[bi];
    }

    TN_FPRINTF(stderr, "[quality] %-8s tets=%lld min-dih=%.2f deg mean=%.2f deg  "
                "hist[<5:%lld 5-10:%lld 10-20:%lld 20-30:%lld 30-40:%lld >40:%lld] "
                "slivers(<10)=%.1f%% degenerate=%lld\n",
                tag, (long long)nt, mn, sum / static_cast<double>(nt),
                h[0], h[1], h[2], h[3], h[4], h[5],
                100.0 * static_cast<double>(h[0] + h[1]) / static_cast<double>(nt), ninv);
}

// Compact tets in place: keep tets t with oldToNew[t]>=0 (new count = nNew),
// remapping tet_neigh through oldToNew and rebuilding point_tet.
static void compact_tets_32(CoarseCDT& m, const std::vector<int>& oldToNew, int nNew) {
    const int64_t nt = m.numTets();
    std::vector<int> tets(static_cast<size_t>(nNew) * 4);
    std::vector<int> label(static_cast<size_t>(nNew));
    std::vector<int> neigh(static_cast<size_t>(nNew) * 4);
    std::vector<unsigned char> mk(static_cast<size_t>(nNew) * 4);

    for (int64_t t = 0; t < nt; ++t) {
        int nT = oldToNew[t];

        if (nT < 0) {
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            tets[4 * nT + i] = m.tets[4 * t + i];
            mk[4 * nT + i]   = m.tet_face_marker[4 * t + i];
            int nb = m.tet_neigh[4 * t + i];
            neigh[4 * nT + i] = (nb >= 0) ? oldToNew[nb] : -1; // freed neighbour -> -1
        }

        label[nT] = m.tet_label[t];
    }

    m.tets.swap(tets);
    m.tet_label.swap(label);
    m.tet_neigh.swap(neigh);
    m.tet_face_marker.swap(mk);
    m.point_tet.assign(static_cast<size_t>(m.numPoints()), -1);

    for (int64_t t = 0; t < m.numTets(); ++t)
        for (int i = 0; i < 4; ++i)
            if (m.point_tet[m.tets[4 * t + i]] < 0) {
                m.point_tet[m.tets[4 * t + i]] = static_cast<int>(t);
            }
}

// Per-tet minimum dihedral angle (deg), computed in parallel. The optimisation
// passes all scan every tet for slivers; b2m_tet_min_dihedral (6 acos per tet) is
// their dominant cost, so each pass computes it once up front.
static void tet_quality_par(const CoarseCDT& m, std::vector<double>& q) {
    const int64_t nt = m.numTets();
    const double* P = m.points.data();
    q.resize(static_cast<size_t>(nt));
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t) {
        const int* tv = &m.tets[4 * t];
        q[static_cast<size_t>(t)] = b2m_tet_min_dihedral(&P[3 * tv[0]], &P[3 * tv[1]], &P[3 * tv[2]], &P[3 * tv[3]]);
    }
}

// Tets around edge (u,v) of tet s, by walking across the two faces of each tet that
// contain the edge (tet_neigh), without a global edge map. Writes up to `cap` tets to
// `out` and returns the ring size if the ring CLOSES within cap tets, else -1 (an
// open ring on the domain boundary, or a larger ring).
static int edge_ring(const CoarseCDT& m, int s, int u, int v, int* out, int cap) {
    int prev = -1, cur = s, n = 0;

    for (int step = 0; step <= cap; ++step) {
        if (n == cap) {
            return -1;
        }

        out[n++] = cur;
        const int* tv = &m.tets[4 * cur];
        int oth[2], k = 0;

        for (int j = 0; j < 4; ++j)
            if (tv[j] != u && tv[j] != v) {
                if (k == 2) {
                    return -1;
                }

                oth[k++] = j;
            }

        if (k != 2) {
            return -1;
        }

        const int nb0 = m.tet_neigh[4 * cur + oth[0]], nb1 = m.tet_neigh[4 * cur + oth[1]];
        const int nxt = (prev < 0 || nb0 != prev) ? nb0 : nb1;

        if (nxt < 0) {
            return -1;
        }

        if (nxt == s) {
            return n;
        }

        prev = cur;
        cur = nxt;
    }

    return -1;
}

// Remove slivers in `m` by 3-2 flips (see the comment above), iterating until a
// pass makes no flip (or max_passes). Parallel form: each pass
// (1) evaluates, in parallel and read-only, the best 3-2 flip around every sliver
// (edge rings found by edge_ring, no global edge map); (2) greedily selects a set of
// flips in order of decreasing gain whose tets AND face-neighbours are pairwise
// disjoint (applying a flip rewrites its outside neighbours' back-pointers); (3)
// applies them serially (O(1) each) and compacts. Keeps tet_neigh/tet_face_marker/
// tet_label consistent; rebuilds point_tet. Returns the number of flips applied.
static int remove_slivers_32(CoarseCDT& m, int max_passes, bool verbose) {
    const double kSliverDeg = 15.0;   // attempt flips around tets worse than this
    const double kGainDeg = 0.5;      // require a strict improvement
    int total = 0;
    std::vector<double> q;

    struct Flip {
        int ot[3];
        int a, b, c, u, v;
        int T0[4], T1[4];
        int lab;
        double gain;
    };

    for (int pass = 0; pass < max_passes; ++pass) {
        const int64_t nt = m.numTets();
        const double* P = m.points.data();
        tet_quality_par(m, q);
        std::vector<Flip> props;

        #pragma omp parallel
        {
            std::vector<Flip> loc;
            #pragma omp for schedule(dynamic, 4096) nowait

            for (int64_t s = 0; s < nt; ++s) {
                const double sq = q[static_cast<size_t>(s)];

                if (sq >= kSliverDeg) {
                    continue;
                }

                const int* sv = &m.tets[4 * s];
                int edges[6][2] = {{sv[0], sv[1]}, {sv[0], sv[2]}, {sv[0], sv[3]}, {sv[1], sv[2]}, {sv[1], sv[3]}, {sv[2], sv[3]}};

                for (int ei = 0; ei < 6; ++ei)
                    if (edges[ei][0] > edges[ei][1]) {
                        std::swap(edges[ei][0], edges[ei][1]);
                    }

                Flip best;
                best.gain = -1.0;

                for (int ei = 0; ei < 6; ++ei) {
                    const int u = edges[ei][0], v = edges[ei][1];
                    int ot[3];

                    if (edge_ring(m, static_cast<int>(s), u, v, ot, 3) != 3) {
                        continue;
                    }

                    const int lab = m.tet_label[ot[0]];

                    if (m.tet_label[ot[1]] != lab || m.tet_label[ot[2]] != lab) {
                        continue;    // crosses an interface
                    }

                    // ring verts = each tet's two verts other than u,v; must be 3 distinct
                    int ring[3], nr = 0;
                    bool okring = true;

                    for (int k = 0; k < 3 && okring; ++k) {
                        const int* cv = &m.tets[4 * ot[k]];

                        for (int j = 0; j < 4; ++j) {
                            const int w = cv[j];

                            if (w == u || w == v) {
                                continue;
                            }

                            bool seen = false;

                            for (int r = 0; r < nr; ++r) if (ring[r] == w) {
                                    seen = true;
                                }

                            if (!seen) {
                                if (nr == 3) {
                                    okring = false;
                                    break;
                                }

                                ring[nr++] = w;
                            }
                        }
                    }

                    if (!okring || nr != 3) {
                        continue;
                    }

                    // the 3 faces containing edge (u,v) disappear: refuse if constrained
                    bool internal_constrained = false;

                    for (int k = 0; k < 3; ++k) {
                        const int* cv = &m.tets[4 * ot[k]];

                        for (int f = 0; f < 4; ++f) {
                            const int f0 = cv[(f + 1) & 3], f1 = cv[(f + 2) & 3], f2 = cv[(f + 3) & 3];
                            const bool hu = (f0 == u || f1 == u || f2 == u), hv = (f0 == v || f1 == v || f2 == v);

                            if (hu && hv && m.tet_face_marker[4 * ot[k] + f]) {
                                internal_constrained = true;
                            }
                        }
                    }

                    if (internal_constrained) {
                        continue;
                    }

                    // canonical order: the same flip proposed from different slivers
                    // (rings walked from different start tets) must be identical
                    std::sort(ring, ring + 3);
                    const int a = ring[0], b = ring[1], c = ring[2];
                    const double ou = b2m_orient3d(&P[3 * a], &P[3 * b], &P[3 * c], &P[3 * u]);
                    const double ov = b2m_orient3d(&P[3 * a], &P[3 * b], &P[3 * c], &P[3 * v]);

                    if (ou == 0.0 || ov == 0.0 || (ou < 0) == (ov < 0)) {
                        continue;    // u,v must be on opposite sides of plane(a,b,c)
                    }

                    double oldq = sq;

                    for (int k = 0; k < 3; ++k) {
                        oldq = std::min(oldq, q[static_cast<size_t>(ot[k])]);
                    }

                    Flip fl;
                    fl.T0[0] = a; fl.T0[1] = b; fl.T0[2] = c; fl.T0[3] = u;
                    fl.T1[0] = a; fl.T1[1] = b; fl.T1[2] = c; fl.T1[3] = v;

                    if (b2m_orient3d(&P[3 * fl.T0[0]], &P[3 * fl.T0[1]], &P[3 * fl.T0[2]], &P[3 * fl.T0[3]]) < 0) {
                        std::swap(fl.T0[0], fl.T0[1]);
                    }

                    if (b2m_orient3d(&P[3 * fl.T1[0]], &P[3 * fl.T1[1]], &P[3 * fl.T1[2]], &P[3 * fl.T1[3]]) < 0) {
                        std::swap(fl.T1[0], fl.T1[1]);
                    }

                    const double q0 = b2m_tet_min_dihedral(&P[3 * fl.T0[0]], &P[3 * fl.T0[1]], &P[3 * fl.T0[2]], &P[3 * fl.T0[3]]);
                    const double q1 = b2m_tet_min_dihedral(&P[3 * fl.T1[0]], &P[3 * fl.T1[1]], &P[3 * fl.T1[2]], &P[3 * fl.T1[3]]);
                    const double gain = std::min(q0, q1) - oldq;

                    if (gain <= kGainDeg || gain <= best.gain) {
                        continue;
                    }

                    // keep the -a cap: the 2 new tets hold the volume of the 3 old ones
                    if (g_opt_guard && lab >= 0 && lab < 6 && g_opt_vol[lab] > 0.0 &&
                            (std::fabs(b2m_orient3d(&P[3 * fl.T0[0]], &P[3 * fl.T0[1]], &P[3 * fl.T0[2]], &P[3 * fl.T0[3]])) / 6.0 >
                             g_opt_vol[lab] ||
                             std::fabs(b2m_orient3d(&P[3 * fl.T1[0]], &P[3 * fl.T1[1]], &P[3 * fl.T1[2]], &P[3 * fl.T1[3]])) / 6.0 >
                             g_opt_vol[lab])) {
                        continue;
                    }

                    for (int k = 0; k < 3; ++k) {
                        fl.ot[k] = ot[k];
                    }

                    fl.a = a;
                    fl.b = b;
                    fl.c = c;
                    fl.u = u;
                    fl.v = v;
                    fl.lab = lab;
                    fl.gain = gain;
                    best = fl;
                }

                if (best.gain > 0) {
                    loc.push_back(best);
                }
            }

            #pragma omp critical
            props.insert(props.end(), loc.begin(), loc.end());
        }

        // deterministic greedy selection: decreasing gain, then smallest tet id
        for (size_t i = 0; i < props.size(); ++i) {
            std::sort(props[i].ot, props[i].ot + 3);
        }

        // total order (thread merge order of `props` is arbitrary): gain, then the
        // tets, then the flipped edge
        std::sort(props.begin(), props.end(), [](const Flip & x, const Flip & y) {
            if (x.gain != y.gain) {
                return x.gain > y.gain;
            }

            for (int k = 0; k < 3; ++k)
                if (x.ot[k] != y.ot[k]) {
                    return x.ot[k] < y.ot[k];
                }

            if (std::min(x.u, x.v) != std::min(y.u, y.v)) {
                return std::min(x.u, x.v) < std::min(y.u, y.v);
            }

            return std::max(x.u, x.v) < std::max(y.u, y.v);
        });

        std::vector<char> claim(static_cast<size_t>(nt), 0), freed(static_cast<size_t>(nt), 0);
        int flips = 0;
        auto tri = [](int x, int y, int z) {
            std::array<int, 3> a3{ { x, y, z } };
            std::sort(a3.begin(), a3.end());
            return a3;
        };

        for (size_t pi = 0; pi < props.size(); ++pi) {
            const Flip& fl = props[pi];
            bool ok = true;

            for (int k = 0; k < 3 && ok; ++k) {
                if (claim[fl.ot[k]]) {
                    ok = false;
                }

                for (int f = 0; f < 4 && ok; ++f) {
                    const int nb = m.tet_neigh[4 * fl.ot[k] + f];

                    if (nb >= 0 && claim[nb]) {
                        ok = false;
                    }
                }
            }

            if (!ok) {
                continue;
            }

            for (int k = 0; k < 3; ++k) {
                claim[fl.ot[k]] = 1;

                for (int f = 0; f < 4; ++f) {
                    const int nb = m.tet_neigh[4 * fl.ot[k] + f];

                    if (nb >= 0) {
                        claim[nb] = 1;
                    }
                }
            }

            // --- apply: map each external face (vertex triple) -> (outside tet, marker)
            struct Ext {
                int nb;
                unsigned char mk;
            };
            std::array<int, 3> ekeys[9];
            Ext evals[9];
            int ne = 0;

            for (int k = 0; k < 3; ++k) {
                const int* cv = &m.tets[4 * fl.ot[k]];

                for (int f = 0; f < 4; ++f) {
                    const int nb = m.tet_neigh[4 * fl.ot[k] + f];

                    if (nb == fl.ot[0] || nb == fl.ot[1] || nb == fl.ot[2]) {
                        continue;    // internal
                    }

                    if (ne < 9) {
                        ekeys[ne] = tri(cv[(f + 1) & 3], cv[(f + 2) & 3], cv[(f + 3) & 3]);
                        evals[ne].nb = nb;
                        evals[ne].mk = m.tet_face_marker[4 * fl.ot[k] + f];
                        ++ne;
                    }
                }
            }

            const int slot[2] = { fl.ot[0], fl.ot[1] };   // reuse 2 slots, free the 3rd
            const int* Tv[2] = { fl.T0, fl.T1 };
            const std::array<int, 3> abc = tri(fl.a, fl.b, fl.c);

            for (int n = 0; n < 2; ++n) {
                const int S = slot[n];

                for (int i = 0; i < 4; ++i) {
                    m.tets[4 * S + i] = Tv[n][i];
                }

                m.tet_label[S] = fl.lab;

                for (int f = 0; f < 4; ++f) {
                    const std::array<int, 3> fa = tri(Tv[n][(f + 1) & 3], Tv[n][(f + 2) & 3], Tv[n][(f + 3) & 3]);

                    if (fa == abc) {                                   // shared (a,b,c) face
                        m.tet_neigh[4 * S + f] = slot[1 - n];
                        m.tet_face_marker[4 * S + f] = 0;
                        continue;
                    }

                    Ext e = { -1, 0 };

                    for (int x = 0; x < ne; ++x) if (ekeys[x] == fa) {
                            e = evals[x];
                            break;
                        }

                    m.tet_neigh[4 * S + f] = e.nb;
                    m.tet_face_marker[4 * S + f] = e.mk;

                    if (e.nb >= 0)                                     // fix outside back-pointer
                        for (int g = 0; g < 4; ++g) {
                            const int* nv = &m.tets[4 * e.nb];

                            if (tri(nv[(g + 1) & 3], nv[(g + 2) & 3], nv[(g + 3) & 3]) == fa) {
                                m.tet_neigh[4 * e.nb + g] = S;
                                break;
                            }
                        }
                }
            }

            freed[fl.ot[2]] = 1;
            ++flips;
        }

        if (flips == 0) {
            break;
        }

        // compact away the freed slots, remapping tet_neigh
        std::vector<int> oldToNew(static_cast<size_t>(nt), -1);
        int w = 0;

        for (int64_t t = 0; t < nt; ++t) if (!freed[t]) {
                oldToNew[t] = w++;
            }

        compact_tets_32(m, oldToNew, w);
        total += flips;

        if (verbose) TN_FPRINTF(stderr, "[sliver] pass %d: %d 3-2 flips -> %lld tets\n",
                                     pass, flips, (long long)m.numTets());
    }

    return total;
}

// Boundary-preserving interior smoothing (smart-Laplacian / optimization).
// Relocates each INTERIOR vertex toward the centroid of its incident-tet
// neighbours, accepting the move only if it raises the worst dihedral of the
// incident patch and inverts no tet. Vertices on a constrained face stay put,
// so the surface (and watertightness) is untouched. This targets "cap" slivers
// whose flat apex is an inserted interior vertex sitting near the surface:
// pulling that apex off the surface fattens the cap. Skips with B2M_NO_SMOOTH=1.
static int smooth_interior(CoarseCDT& m, int passes, bool verbose) {
    const auto st0 = std::chrono::steady_clock::now();
    auto sms = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st0).count();
    };
    double s_setup = 0, s_q = 0, s_cand = 0, s_col = 0, s_move = 0, s_dirty = 0, s_prev = 0;
    auto slap = [&](double& acc) {
        const double t = sms();
        acc += t - s_prev;
        s_prev = t;
    };
    // Parallel form: the topology is fixed during smoothing, so each pass (1) finds
    // the candidate vertices (interior, patch min-dihedral < 25) in parallel, (2)
    // greedily colours them so that no two vertices of one colour share a tet (a
    // move only reads/changes its own incident tets), and (3) moves each colour's
    // vertices in parallel. Colours are processed in order, so this is a
    // deterministic multi-colour Gauss-Seidel sweep.
    const int nv = static_cast<int>(m.numPoints());
    const int nt = static_cast<int>(m.numTets());
    std::vector<char> bnd(static_cast<size_t>(nv), 0);
    std::vector<int> vs(static_cast<size_t>(nv) + 1, 0), vt(static_cast<size_t>(nt) * 4);

    for (int t = 0; t < nt; ++t) {
        const int* tv = &m.tets[4 * t];

        for (int lv = 0; lv < 4; ++lv) {
            vs[tv[lv] + 1]++;
        }

        for (int f = 0; f < 4; ++f)            // face f = the 3 vertices != f
            if (m.tet_face_marker[4 * t + f])
                for (int lv = 0; lv < 4; ++lv)
                    if (lv != f) {
                        bnd[tv[lv]] = 1;
                    }
    }

    for (int v = 0; v < nv && v < static_cast<int>(m.point_marker.size()); ++v) {
        bnd[v] |= m.point_marker[v];   // trussnet: interface / junction / corner nodes never move
    }

    for (int v = 0; v < nv; ++v) {
        vs[v + 1] += vs[v];
    }

    {
        std::vector<int> fill(vs.begin(), vs.end() - 1);

        for (int t = 0; t < nt; ++t)
            for (int lv = 0; lv < 4; ++lv) {
                vt[fill[m.tets[4 * t + lv]]++] = t;
            }
    }

    // signed volume*6 of tet t, optionally with local vertex `lvm` moved to `pv`.
    auto vol6 = [&](int t, const double * pv, int lvm) -> double {
        const int* tv = &m.tets[4 * t];
        const double* q[4];

        for (int k = 0; k < 4; ++k) {
            q[k] = (k == lvm) ? pv : &m.points[3 * tv[k]];
        }

        double a[3], b[3], c[3];

        for (int k = 0; k < 3; ++k) {
            a[k] = q[1][k] - q[0][k];
            b[k] = q[2][k] - q[0][k];
            c[k] = q[3][k] - q[0][k];
        }

        return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
        a[2] * (b[0] * c[1] - b[1] * c[0]);
    };
    auto patch_min_dih = [&](int v, const double * pos) -> double {
        double mn = 180.0;

        for (int i = vs[v]; i < vs[v + 1]; ++i) {
            const int* tv = &m.tets[4 * vt[i]];
            const double* q[4];

            for (int k = 0; k < 4; ++k) {
                q[k] = (tv[k] == v) ? pos : &m.points[3 * tv[k]];
            }

            mn = std::fmin(mn, b2m_tet_min_dihedral(q[0], q[1], q[2], q[3]));
        }

        return mn;
    };
    // Only DIRTY vertices are examined: after a pass, the ones that moved or share a
    // tet with a moved vertex. Any other vertex has an unchanged patch, so its
    // (deterministic) move attempt would fail again -- skipping it is exact.
    std::vector<char> dirty(static_cast<size_t>(nv), 1), movedf(static_cast<size_t>(nv), 0);
    // trussnet: a vertex whose move failed on an unchanged patch (same incident
    // tets, same node positions) fails again -- remembered across calls
    if (m.point_failed.size() != static_cast<size_t>(nv)) {
        m.point_failed.assign(static_cast<size_t>(nv), 0);
        m.point_sig.assign(static_cast<size_t>(nv), 0);
    }

    auto patch_sig = [&](int v) {
        uint64_t hsh = 1469598103934665603ULL ^ static_cast<uint64_t>(vs[v + 1] - vs[v]);

        for (int i = vs[v]; i < vs[v + 1]; ++i) {
            uint64_t ht = 0;
            const int* tv = &m.tets[4 * vt[i]];

            for (int k = 0; k < 4; ++k) {   // order-independent per tet, then summed
                uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(tv[k])) * 0x9E3779B97F4A7C15ULL;
                const double* q = &m.points[3 * tv[k]];
                uint64_t b[3];
                std::memcpy(b, q, sizeof(b));
                x ^= b[0] * 0xC2B2AE3D27D4EB4FULL ^ (b[1] << 1) * 0x165667B19E3779F9ULL ^ (b[2] << 2) * 0x27D4EB2F165667C5ULL;
                ht += x ^ (x >> 29);
            }

            hsh += ht * 0xFF51AFD7ED558CCDULL;
        }

        return hsh;
    };
    // try to move v toward the centroid of its patch; returns 1 if moved
    auto try_move = [&](int v) -> int {
        const double cur0[3] = { m.points[3 * v], m.points[3 * v + 1], m.points[3 * v + 2] };
        const double cur = patch_min_dih(v, cur0);

        if (cur >= 25.0) {
            return 0;    // patch already well-shaped
        }

        double cen[3] = { 0, 0, 0 };
        int cnt = 0;

        for (int i = vs[v]; i < vs[v + 1]; ++i) {
            const int* tv = &m.tets[4 * vt[i]];

            for (int k = 0; k < 4; ++k)
                if (tv[k] != v) {
                    cen[0] += m.points[3 * tv[k]];
                    cen[1] += m.points[3 * tv[k] + 1];
                    cen[2] += m.points[3 * tv[k] + 2];
                    ++cnt;
                }
        }

        if (!cnt) {
            return 0;
        }

        cen[0] /= cnt;
        cen[1] /= cnt;
        cen[2] /= cnt;
        double best = cur, bestpos[3] = { cur0[0], cur0[1], cur0[2] };
        static const double alphas[3] = { 1.0, 0.6, 0.3 };
        const int deg = vs[v + 1] - vs[v];
        std::vector<signed char> sgn(static_cast<size_t>(deg));
        std::vector<signed char> lv(static_cast<size_t>(deg));

        for (int i = 0; i < deg; ++i) {
            const int t = vt[vs[v] + i];
            const int* tv = &m.tets[4 * t];
            sgn[i] = vol6(t, nullptr, -1) > 0 ? 1 : 0;
            lv[i] = -1;

            for (int k = 0; k < 4; ++k) if (tv[k] == v) {
                    lv[i] = static_cast<signed char>(k);
                }
        }

        for (int ai = 0; ai < 3; ++ai) {
            const double a = alphas[ai];
            const double np[3] = { cur0[0] + a * (cen[0] - cur0[0]),
                                   cur0[1] + a * (cen[1] - cur0[1]),
                                   cur0[2] + a * (cen[2] - cur0[2])
                                 };
            bool ok = true;

            for (int i = 0; i < deg && ok; ++i) {
                const double nvg = vol6(vt[vs[v] + i], np, lv[i]);

                if ((nvg > 0) != (sgn[i] != 0) || std::fabs(nvg) < 1e-12) {
                    ok = false;    // inverted or degenerate
                }

                // keep the -a cap: a move may not grow a tet past its tissue's max volume
                if (ok && g_opt_guard) {
                    const int lb = m.tet_label[vt[vs[v] + i]];
                    const double vc = (lb >= 0 && lb < 6) ? g_opt_vol[lb] : 0.0;

                    if (vc > 0.0 && std::fabs(nvg) / 6.0 > vc) {
                        ok = false;
                    }
                }
            }

            if (!ok) {
                continue;
            }

            const double qn = patch_min_dih(v, np);

            if (qn > best) {
                best = qn;
                bestpos[0] = np[0];
                bestpos[1] = np[1];
                bestpos[2] = np[2];
            }
        }

        if (best > cur + 1e-6) {
            m.points[3 * v] = bestpos[0];
            m.points[3 * v + 1] = bestpos[1];
            m.points[3 * v + 2] = bestpos[2];
            movedf[v] = 1;
            return 1;
        }

        return 0;
    };

    int moved = 0;
    std::vector<double> qtet;
    std::vector<char> iscand(static_cast<size_t>(nv), 0);
    std::vector<int> color(static_cast<size_t>(nv), -1);
    slap(s_setup);

    for (int pass = 0; pass < passes; ++pass) {
        // (1) candidates: patch min-dihedral from a per-tet quality array (each tet
        // evaluated once, not once per vertex)
        if (pass == 0) {
            tet_quality_par(m, qtet);
        }

        slap(s_q);

        #pragma omp parallel for schedule(dynamic, 2048)

        for (int v = 0; v < nv; ++v) {
            iscand[v] = 0;

            if (!dirty[v] || bnd[v] || vs[v] == vs[v + 1] || (m.point_failed[v] && m.point_sig[v] == patch_sig(v))) {
                continue;
            }

            double mn = 180.0;

            if (pass == 0) {
                for (int i = vs[v]; i < vs[v + 1]; ++i) {
                    mn = std::fmin(mn, qtet[static_cast<size_t>(vt[i])]);
                }
            } else {
                const double c0[3] = { m.points[3 * v], m.points[3 * v + 1], m.points[3 * v + 2] };
                mn = patch_min_dih(v, c0);
            }

            if (mn < 25.0) {
                iscand[v] = 1;
            }
        }

            slap(s_cand);
            // (2) greedy colouring of the candidates (neighbours = vertices sharing a tet)
        std::vector<std::vector<int>> byColor;

        for (int v = 0; v < nv; ++v) {
            color[v] = -1;
        }

        for (int v = 0; v < nv; ++v) {
            if (!iscand[v]) {
                continue;
            }

            uint64_t usedmask = 0;
            std::vector<char> usedbig;

            for (int i = vs[v]; i < vs[v + 1]; ++i) {
                const int* tv = &m.tets[4 * vt[i]];

                for (int k = 0; k < 4; ++k) {
                    const int w = tv[k];

                    if (w == v || color[w] < 0) {
                        continue;
                    }

                    if (color[w] < 64) {
                        usedmask |= (1ULL << color[w]);
                    } else {
                        if (usedbig.size() <= static_cast<size_t>(color[w])) {
                            usedbig.resize(static_cast<size_t>(color[w]) + 1, 0);
                        }

                        usedbig[color[w]] = 1;
                    }
                }
            }

            int c = 0;

            while (c < 64 && (usedmask >> c & 1ULL)) {
                ++c;
            }

            if (c == 64)
                while (static_cast<size_t>(c) < usedbig.size() && usedbig[c]) {
                    ++c;
                }

            color[v] = c;

            if (byColor.size() <= static_cast<size_t>(c)) {
                byColor.resize(static_cast<size_t>(c) + 1);
            }

            byColor[c].push_back(v);
        }

            slap(s_col);
            // (3) move colour by colour
        int pmoved = 0;

        for (size_t c = 0; c < byColor.size(); ++c) {
            const std::vector<int>& vc = byColor[c];
            const int n = static_cast<int>(vc.size());
            #pragma omp parallel for schedule(dynamic, 256) reduction(+:pmoved)

            for (int i = 0; i < n; ++i) {
                const int v = vc[i];
                const int r = try_move(v);
                pmoved += r;
                m.point_failed[v] = r ? 0 : 1;

                if (!r) {
                    m.point_sig[v] = patch_sig(v);
                }
            }
        }

        moved += pmoved;

        slap(s_move);
        // next pass: dirty = moved vertices and their tet-neighbours
        std::fill(dirty.begin(), dirty.end(), 0);

        for (int v = 0; v < nv; ++v) {
            if (!movedf[v]) {
                continue;
            }

            movedf[v] = 0;

            for (int i = vs[v]; i < vs[v + 1]; ++i) {
                const int* tv = &m.tets[4 * vt[i]];

                for (int k = 0; k < 4; ++k) {
                    dirty[tv[k]] = 1;
                }
            }
        }

    
        slap(s_dirty);

        if (verbose) {
            TN_FPRINTF(stderr, "[smooth] pass %d: moved %d interior verts (%zu colours)\n", pass, pmoved,
                        byColor.size());
        }

        if (!pmoved) {
            break;
        }
    }

    if (std::getenv("TN_OPT_PROFILE")) {
        TN_FPRINTF(stderr, "[smooth] setup %.0f, quality %.0f, candidates %.0f, colouring %.0f, moves %.0f, dirty %.0f ms\n",
                   s_setup, s_q, s_cand, s_col, s_move, s_dirty);
    }

    return moved;
}

// Recompute constrained-face markers from adjacency + labels: a face is a
// boundary/interface to protect iff it has no neighbour (exterior) or the
// neighbour has a different label. The final mesh faces are derived from labels
// independently (b2m_pipeline.cpp), so tet_face_marker is used only inside this
// optimization phase -- recomputing it after a topology change is exact and lets
// the flip/collapse passes skip per-face marker bookkeeping.
static void recompute_face_markers(CoarseCDT& m) {
    const int64_t nt = m.numTets();
    m.tet_face_marker.assign(static_cast<size_t>(nt) * 4, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t)
        for (int f = 0; f < 4; ++f) {
            int nb = m.tet_neigh[4 * t + f];

            if (nb < 0 || m.tet_label[nb] != m.tet_label[t]) {
                m.tet_face_marker[4 * t + f] = 1;
            }
        }
}

// 2->3 flip pass: across an interior face shared by two same-label tets (apexes
// d, e), replace the 2 tets by 3 sharing edge d-e -- when the union is convex
// (all 3 sub-tets proper) and it raises the worst min-dihedral. Constrained
// (boundary/interface) faces are never flipped, so the surface is preserved.
static int flip_23(CoarseCDT& m, bool verbose) {
    const double kGain = 1.0;      // require >= 1 deg min-dihedral improvement
    const double kSliver = 18.0;   // only act when a tet is sliver-ish
    int total = 0;

    for (int pass = 0; pass < 4; ++pass) {
        const int64_t nt = m.numTets();
        const double* P = m.points.data();
        std::vector<char> touched(nt, 0), dead(nt, 0);
        std::vector<std::array<int, 4>> newtets;
        std::vector<int> newlab;
        int flips = 0;
        std::vector<double> qt;
        tet_quality_par(m, qt);   // tets are not modified in place here (new ones appended)

        for (int64_t t = 0; t < nt; ++t) {
            if (touched[t]) {
                continue;
            }

            const int* tv = &m.tets[4 * t];
            const double dq_t = qt[static_cast<size_t>(t)];

            if (dq_t >= kSliver) {
                // both tets of a flippable pair must include a sliver-ish one; the
                // pair is also visited from the neighbour, so skipping is exact
                bool anybad = false;

                for (int f = 0; f < 4; ++f) {
                    const int nb = m.tet_neigh[4 * t + f];

                    if (nb > (int)t && qt[static_cast<size_t>(nb)] < kSliver) {
                        anybad = true;
                    }
                }

                if (!anybad) {
                    continue;
                }
            }

            for (int f = 0; f < 4; ++f) {
                int nb = m.tet_neigh[4 * t + f];

                if (nb < 0 || nb <= (int)t || touched[nb]) {
                    continue;    // interior face, once, both free
                }

                if (m.tet_face_marker[4 * t + f]) {
                    continue;    // constrained -> never flip
                }

                int d = tv[f], a = tv[(f + 1) & 3], b = tv[(f + 2) & 3], c = tv[(f + 3) & 3];
                const int* nv = &m.tets[4 * nb];
                int e = -1;

                for (int k = 0; k < 4; ++k)
                    if (nv[k] != a && nv[k] != b && nv[k] != c) {
                        e = nv[k];
                        break;
                    }

                if (e < 0) {
                    continue;
                }

                const double dq_n = qt[static_cast<size_t>(nb)];
                double oldq = std::fmin(dq_t, dq_n);

                if (oldq >= kSliver) {
                    continue;
                }

                int A = a, B = b, C = c;        // order so orient3d(A,B,C,d) > 0

                if (b2m_orient3d(&P[3 * A], &P[3 * B], &P[3 * C], &P[3 * d]) < 0) {
                    std::swap(A, B);
                }

                double o1 = b2m_orient3d(&P[3 * A], &P[3 * B], &P[3 * d], &P[3 * e]);
                double o2 = b2m_orient3d(&P[3 * B], &P[3 * C], &P[3 * d], &P[3 * e]);
                double o3 = b2m_orient3d(&P[3 * C], &P[3 * A], &P[3 * d], &P[3 * e]);

                if (!(o1 > 0 && o2 > 0 && o3 > 0)) {
                    continue;    // non-convex union -> invalid flip
                }

                double q1 = b2m_tet_min_dihedral(&P[3 * A], &P[3 * B], &P[3 * d], &P[3 * e]);
                double q2 = b2m_tet_min_dihedral(&P[3 * B], &P[3 * C], &P[3 * d], &P[3 * e]);
                double q3 = b2m_tet_min_dihedral(&P[3 * C], &P[3 * A], &P[3 * d], &P[3 * e]);

                if (std::fmin(q1, std::fmin(q2, q3)) <= oldq + kGain) {
                    continue;
                }

                int lab = m.tet_label[t];
                newtets.push_back({ A, B, d, e });
                newlab.push_back(lab);
                newtets.push_back({ B, C, d, e });
                newlab.push_back(lab);
                newtets.push_back({ C, A, d, e });
                newlab.push_back(lab);
                dead[t] = dead[nb] = touched[t] = touched[nb] = 1;
                ++flips;
                break;
            }
        }

        if (!flips) {
            break;
        }

        for (size_t i = 0; i < newtets.size(); ++i) {
            for (int k = 0; k < 4; ++k) {
                m.tets.push_back(newtets[i][k]);
                m.tet_neigh.push_back(-1);
                m.tet_face_marker.push_back(0);
            }

            m.tet_label.push_back(newlab[i]);
        }

        dead.resize(static_cast<size_t>(m.numTets()), 0);
        compact_dead_cpu(m, dead);       // rebuilds tet_neigh from scratch
        recompute_face_markers(m);
        total += flips;

        if (verbose) TN_FPRINTF(stderr, "[flip23] pass %d: %d flips -> %lld tets\n",
                                     pass, flips, (long long)m.numTets());
    }

    return total;
}

// Drop vertices no tet references (e.g. after edge collapses), remapping tets,
// point_marker and point_tet so the exported node list has no orphans.
static void compact_points(CoarseCDT& m) {
    const int64_t np = m.numPoints();
    const int64_t nt = m.numTets();
    std::vector<int> remap(static_cast<size_t>(np), -1);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < 4 * nt; ++i) {
        remap[m.tets[i]] = 0;    // benign same-value write
    }

    int w = 0;
    std::vector<double> pts;
    pts.reserve(m.points.size());
    std::vector<unsigned char> pm;
    std::vector<int> ptet, po;
    std::vector<unsigned char> pf;
    std::vector<uint64_t> ps;
    const bool have_pm = (int64_t)m.point_marker.size() == np;
    const bool have_po = (int64_t)m.point_orig.size() == np;
    const bool have_pf = (int64_t)m.point_failed.size() == np && (int64_t)m.point_sig.size() == np;

    for (int64_t v = 0; v < np; ++v)
        if (remap[v] == 0) {
            remap[v] = w++;
            pts.push_back(m.points[3 * v]);
            pts.push_back(m.points[3 * v + 1]);
            pts.push_back(m.points[3 * v + 2]);

            if (have_pm) {
                pm.push_back(m.point_marker[v]);
            }

            if (have_po) {
                po.push_back(m.point_orig[v]);
            }

            if (have_pf) {
                pf.push_back(m.point_failed[v]);
                ps.push_back(m.point_sig[v]);
            }
        }

    if (w == np) {
        return;    // nothing orphaned
    }

    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < 4 * nt; ++i) {
        m.tets[i] = remap[m.tets[i]];
    }

    m.points.swap(pts);

    if (have_pm) {
        m.point_marker.swap(pm);
    }

    if (have_po) {
        m.point_orig.swap(po);
    }

    if (have_pf) {
        m.point_failed.swap(pf);
        m.point_sig.swap(ps);
    }

    m.point_tet.assign(static_cast<size_t>(w), -1);   // rebuilt by adjacency users if needed
}

// Edge-collapse pass (TetGen CombineImprove): collapse an interior vertex d onto
// a neighbour x along a short edge of a sliver, deleting the tets on that edge.
// Refused if d is on the boundary, if a deleted tet carries a boundary/interface
// face (would tear the surface), or if any surviving incident tet would invert.
static int collapse_interior(CoarseCDT& m, bool verbose) {
    const auto ct0 = std::chrono::steady_clock::now();
    auto cms = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ct0).count();
    };
    const double kSliver = 18.0;
    const int64_t nt = m.numTets();
    const int nv = static_cast<int>(m.numPoints());
    std::vector<char> bnd(nv, 0);
    // vertex -> tets: flat CSR for the initial mesh + a small overflow list for the
    // tets a vertex gains through collapses in this pass (a vector per vertex was
    // the dominant cost on large meshes)
    std::vector<int> vs(static_cast<size_t>(nv) + 1, 0), vt(static_cast<size_t>(nt) * 4);
    std::unordered_map<int, std::vector<int>> vextra;

    for (int64_t t = 0; t < nt; ++t) {
        const int* tv = &m.tets[4 * t];

        for (int lv = 0; lv < 4; ++lv) {
            vs[tv[lv] + 1]++;
        }

        for (int f = 0; f < 4; ++f)
            if (m.tet_face_marker[4 * t + f])
                for (int lv = 0; lv < 4; ++lv)
                    if (lv != f) {
                        bnd[tv[lv]] = 1;
                    }
    }

    // (collapse: no trussnet type freeze -- a collapse removes a node and moves
    // none, so a typed node that is on no constrained face may go; smoothing keeps
    // the freeze, since a moved node's label set would be stale)

    for (int v = 0; v < nv; ++v) {
        vs[v + 1] += vs[v];
    }

    {
        std::vector<int> fill(vs.begin(), vs.end() - 1);

        for (int64_t t = 0; t < nt; ++t)
            for (int lv = 0; lv < 4; ++lv) {
                vt[fill[m.tets[4 * t + lv]]++] = static_cast<int>(t);
            }
    }

    std::vector<int> v2tbuf;
    auto v2t_of = [&](int v) -> const std::vector<int>& {
        v2tbuf.assign(vt.begin() + vs[v], vt.begin() + vs[v + 1]);
        std::unordered_map<int, std::vector<int>>::const_iterator it = vextra.find(v);

        if (it != vextra.end()) {
            v2tbuf.insert(v2tbuf.end(), it->second.begin(), it->second.end());
        }

        return v2tbuf;
    };

    const double* P = m.points.data();
    std::vector<char> dead(static_cast<size_t>(nt), 0), removedv(nv, 0), modified(static_cast<size_t>(nt), 0);
    std::vector<double> qt;
    tet_quality_par(m, qt);   // stale only for tets rewritten below (recomputed then)
    auto qual = [&](int tt) -> double {
        if (!modified[tt]) {
            return qt[static_cast<size_t>(tt)];
        }

        const int* q = &m.tets[4 * tt];
        return b2m_tet_min_dihedral(&P[3 * q[0]], &P[3 * q[1]], &P[3 * q[2]], &P[3 * q[3]]);
    };
    auto vol = [&](int A, int B, int C, int D) {
        return b2m_orient3d(&P[3 * A], &P[3 * B], &P[3 * C], &P[3 * D]);
    };
    // trussnet: two phases. (1) parallel, read-only on the unmodified mesh: the
    // best collapse d->x per sliver-ish tet (same validity / gain rules as the
    // serial gpu_brain2mesh pass); (2) serial commit by decreasing gain, skipping a
    // proposal whose star of d was touched by an earlier commit (it is re-proposed
    // next round). The evaluation dominated the serial pass.
    struct Prop {
        double gain;
        int t, d, x;
    };
    std::vector<std::vector<Prop>> per;
#ifdef _OPENMP
    per.resize(static_cast<size_t>(omp_get_max_threads()));
#else
    per.resize(1);
#endif
    #pragma omp parallel
    {
#ifdef _OPENMP
        std::vector<Prop>& mine = per[static_cast<size_t>(omp_get_thread_num())];
#else
        std::vector<Prop>& mine = per[0];
#endif
        #pragma omp for schedule(dynamic, 1024)

        for (int64_t t = 0; t < nt; ++t) {
            if (qt[static_cast<size_t>(t)] >= kSliver) {
                continue;
            }

            const int* tv = &m.tets[4 * t];
            Prop best = { 0.0, -1, -1, -1 };

            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    if (i == j) {
                        continue;
                    }

                    const int d = tv[i], x = tv[j];

                    if (bnd[d] || d == x) {
                        continue;
                    }

                    bool ok = true;
                    double before = 180.0, after = 180.0;

                    for (int s2 = vs[d]; s2 < vs[d + 1] && ok; ++s2) {   // tets at d
                        const int tt = vt[s2];
                        const int* q = &m.tets[4 * tt];
                        const bool hasx = (q[0] == x || q[1] == x || q[2] == x || q[3] == x);
                        before = std::fmin(before, qt[static_cast<size_t>(tt)]);

                        if (hasx) {   // dies: refuse if it carries a boundary face opposite d
                            for (int k = 0; k < 4; ++k)
                                if (q[k] == d && m.tet_face_marker[4 * tt + k]) {
                                    ok = false;
                                }

                            continue;
                        }

                        int w[4];

                        for (int k = 0; k < 4; ++k) {
                            w[k] = (q[k] == d) ? x : q[k];
                        }

                        if (vol(w[0], w[1], w[2], w[3]) <= 1e-12) {
                            ok = false;    // inverted / degenerate
                            break;
                        }

                        if (g_opt_guard && b2m_check_bad(&P[3 * w[0]], &P[3 * w[1]], &P[3 * w[2]], &P[3 * w[3]], g_opt_q,
                                                         0.0, 0.0, 0.0)) {
                            ok = false;    // would create a radius-edge-bad tet
                            break;
                        }

                        after = std::fmin(after, b2m_tet_min_dihedral(&P[3 * w[0]], &P[3 * w[1]], &P[3 * w[2]], &P[3 * w[3]]));
                    }

                    if (ok && after > before + 1.0 && after - before > best.gain) {
                        best = { after - before, static_cast<int>(t), d, x };
                    }
                }

            if (best.t >= 0) {
                mine.push_back(best);
            }
        }
    }
    const double c_eval = cms();
    std::vector<Prop> props;

    for (auto& v : per) {
        props.insert(props.end(), v.begin(), v.end());
    }

    std::sort(props.begin(), props.end(), [](const Prop& a, const Prop& b) {
        return a.gain > b.gain || (a.gain == b.gain && (a.t < b.t || (a.t == b.t && (a.d < b.d || (a.d == b.d && a.x < b.x)))));
    });
    int collapses = 0;

    for (const Prop& pr : props) {
        const int d = pr.d, x = pr.x;

        if (removedv[d] || removedv[x] || dead[pr.t]) {
            continue;
        }

        bool fresh = true;   // the star of d must be as evaluated

        for (int s2 = vs[d]; s2 < vs[d + 1] && fresh; ++s2) {
            fresh = !dead[vt[s2]] && !modified[vt[s2]];
        }

        if (!fresh || vextra.count(d)) {
            continue;
        }

        for (int s2 = vs[d]; s2 < vs[d + 1]; ++s2) {
            const int tt = vt[s2];
            int* q = &m.tets[4 * tt];
            const bool hasx = (q[0] == x || q[1] == x || q[2] == x || q[3] == x);

            if (hasx) {
                dead[tt] = 1;
            } else {
                for (int k = 0; k < 4; ++k)
                    if (q[k] == d) {
                        q[k] = x;
                    }

                modified[tt] = 1;
                vextra[x].push_back(tt);
            }
        }

        removedv[d] = 1;
        ++collapses;
    }

    (void)qual;
    (void)v2t_of;

    const double c_commit = cms();

    if (collapses) {
        compact_dead_cpu(m, dead);
        compact_points(m);
        recompute_face_markers(m);

        if (std::getenv("TN_OPT_PROFILE")) {
            TN_FPRINTF(stderr, "[collapse] setup+eval %.0f ms, commit %.0f ms, rebuild %.0f ms\n", c_eval, c_commit - c_eval,
                       cms() - c_commit);
        }

        if (verbose) TN_FPRINTF(stderr, "[collapse] %d edge collapses -> %lld tets\n",
                                     collapses, (long long)m.numTets());
    }


    return collapses;
}

// Steiner insertion (TetGen add_steinerpt_to_repair): break a residual interior
// sliver S by inserting a point p at the barycenter of its fattest same-label
// neighbour F (so p sits OFF S's flat plane), carving the 2-tet cavity {S,F} and
// connecting p to the cavity's 6 boundary faces. Applied only when p sees every
// boundary face (star-shaped -> all new tets proper) and the worst min-dihedral
// strictly improves. Single-label cavity + boundary faces kept as faces of the
// new tets => the surface/interfaces are preserved. Gated by B2M_NO_STEINER.
// Kite flattening (trussnet): the worst tets left after the flips are flat
// "kites" whose four nodes all lie on one interface a|b -- every face is
// constrained, so no flip / collapse may touch them. Relabelling such a tet to
// the label of most of its neighbours (allowed when all four nodes carry that
// label, as a kite on a|b does) moves the interface by the kite's near-zero
// volume and unconstrains its faces, so the next flip pass can remove it. A kite
// on the exterior surface (all nodes carry label 0, >= 2 exterior faces) is
// deleted instead. Taken only if it strictly reduces the tet's constrained faces;
// serial from the worst tet up, so neighbouring kites see each other's changes.
// `lset(v)` returns the up-to-4 labels of point v (-1 padded).
template <class LSet>
static int flatten_kites(CoarseCDT& m, double qmax, const LSet& lset, bool verbose) {
    const int64_t nt = m.numTets();
    std::vector<double> q;
    tet_quality_par(m, q);
    std::vector<int> cand;

    for (int64_t t = 0; t < nt; ++t)
        if (q[t] < qmax) {
            cand.push_back(static_cast<int>(t));
        }

    std::sort(cand.begin(), cand.end(), [&](int a, int b) {
        return q[a] < q[b] || (q[a] == q[b] && a < b);
    });
    std::vector<char> dead(static_cast<size_t>(nt), 0);
    int nrel = 0, ndel = 0;
    auto nb_label = [&](int t, int f) {
        const int nb = m.tet_neigh[4 * t + f];
        return nb < 0 || dead[nb] ? 0 : m.tet_label[nb];   // 0 = exterior
    };

    for (int t : cand) {
        const int cur = m.tet_label[t];
        int labs[4], cnt[4], nl = 0, ncons = 0, next = 0;

        for (int f = 0; f < 4; ++f) {
            const int l = nb_label(t, f);
            next += l == 0 && (m.tet_neigh[4 * t + f] < 0 || dead[m.tet_neigh[4 * t + f]]);
            ncons += l != cur;

            if (l == cur) {
                continue;
            }

            int k = 0;

            while (k < nl && labs[k] != l) {
                ++k;
            }

            if (k == nl) {
                labs[nl] = l;
                cnt[nl++] = 0;
            }

            ++cnt[k];
        }

        if (ncons == 0) {
            continue;
        }

        // candidate labels by how many faces they share, most first
        for (int x = 0; x < nl; ++x)
            for (int y = x + 1; y < nl; ++y)
                if (cnt[y] > cnt[x] || (cnt[y] == cnt[x] && labs[y] < labs[x])) {
                    std::swap(cnt[x], cnt[y]);
                    std::swap(labs[x], labs[y]);
                }

        int best = -1;

        for (int k = 0; k < nl && best < 0; ++k) {
            const int L = labs[k], after = 4 - cnt[k];

            if (after > ncons || (L == 0 && (after == ncons || next < 2))) {
                continue;
            }

            bool ok = true;   // every node must carry the new label

            for (int e = 0; e < 4 && ok; ++e) {
                const auto ls = lset(m.tets[4 * t + e]);
                ok = ls[0] == L || ls[1] == L || ls[2] == L || ls[3] == L;
            }

            if (!ok) {
                continue;
            }

            if (after == ncons) {
                // as many constrained faces (a flat 2|2 kite on the interface):
                // relabelling it is an edge flip of the interface -- keep it only if
                // it opens a 3-2 flip, i.e. an edge of t whose ring is exactly 3 tets
                // all labelled L (so no constrained face contains the edge)
                m.tet_label[t] = L;
                bool opens = false;

                for (int a = 0; a < 4 && !opens; ++a)
                    for (int b = a + 1; b < 4 && !opens; ++b) {
                        int ring[8];
                        const int nr = edge_ring(m, t, m.tets[4 * t + a], m.tets[4 * t + b], ring, 8);

                        if (nr == 3) {
                            opens = !dead[ring[0]] && !dead[ring[1]] && !dead[ring[2]] && m.tet_label[ring[0]] == L &&
                                    m.tet_label[ring[1]] == L && m.tet_label[ring[2]] == L;
                        }
                    }

                m.tet_label[t] = cur;

                if (!opens) {
                    continue;
                }
            }

            best = L;
        }

        if (best < 0) {
            continue;
        }

        if (best == 0) {
            dead[t] = 1;
            ++ndel;
        } else {
            m.tet_label[t] = best;
            ++nrel;
        }
    }

    if (ndel > 0) {
        compact_dead_cpu(m, dead);
    }

    if (ndel > 0 || nrel > 0) {
        recompute_face_markers(m);
    }

    if (verbose) {
        TN_FPRINTF(stderr, "[opt] kites: %d relabelled, %d exterior ones deleted (of %zu below %.0f deg)\n", nrel, ndel,
                   cand.size(), qmax);
    }

    return nrel + ndel;
}

static int insert_steiner_slivers(CoarseCDT& m, bool verbose) {
    const double kSliver = 12.0;   // only the stubborn residual slivers
    const double kGain = 2.0;
    const int64_t nt = m.numTets();
    const double* P = m.points.data();
    std::vector<char> touched(static_cast<size_t>(nt), 0), dead(static_cast<size_t>(nt), 0);

    struct Ins {
        double p[3];
        int s, F;
        std::array<int, 3> faces[6];   // boundary faces (oriented for orient3d>0 with p)
        int nf;
    };
    std::vector<Ins> ins;

    std::vector<double> qt;
    tet_quality_par(m, qt);   // tets are only killed + appended here, never rewritten
    auto md4 = [&](int a, int b, int c, const double * d) {
        return b2m_tet_min_dihedral(&P[3 * a], &P[3 * b], &P[3 * c], d);
    };

    for (int64_t s = 0; s < nt; ++s) {
        if (touched[s]) {
            continue;
        }

        const int* sv = &m.tets[4 * s];
        double oldq = qt[static_cast<size_t>(s)];

        if (oldq >= kSliver) {
            continue;
        }

        // fattest same-label interior face-neighbour F
        int F = -1, lab = m.tet_label[s];
        double bestq = oldq;

        for (int f = 0; f < 4; ++f) {
            if (m.tet_face_marker[4 * s + f]) {
                continue;    // interface/exterior face -> keep cavity single-label
            }

            int nb = m.tet_neigh[4 * s + f];

            if (nb < 0 || touched[nb] || m.tet_label[nb] != lab) {
                continue;
            }

            const int* fv = &m.tets[4 * nb];
            double q = qt[static_cast<size_t>(nb)];

            if (q > bestq) {
                bestq = q;
                F = nb;
            }
        }

        if (F < 0) {
            continue;
        }

        const int* fv = &m.tets[4 * F];
        double p[3] = { 0, 0, 0 };

        for (int k = 0; k < 4; ++k) {
            p[0] += P[3 * fv[k]];
            p[1] += P[3 * fv[k] + 1];
            p[2] += P[3 * fv[k] + 2];
        }

        p[0] *= 0.25;
        p[1] *= 0.25;
        p[2] *= 0.25;

        // boundary faces of {s,F}: every face of s and F except the shared one
        Ins cand;
        cand.s = (int)s;
        cand.F = F;
        cand.nf = 0;
        bool ok = true;
        double newq = 180.0;
        int cav[2] = { (int)s, F };

        for (int ci = 0; ci < 2 && ok; ++ci) {
            int t = cav[ci];
            const int* tv = &m.tets[4 * t];

            for (int f = 0; f < 4; ++f) {
                if (m.tet_neigh[4 * t + f] == cav[1 - ci]) {
                    continue;    // shared face (internal to cavity)
                }

                int a = tv[(f + 1) & 3], b = tv[(f + 2) & 3], c = tv[(f + 3) & 3], w = tv[f];
                // p must lie on the same side of (a,b,c) as the cavity (where w is)
                double sw = b2m_orient3d(&P[3 * a], &P[3 * b], &P[3 * c], &P[3 * w]);
                double sp = b2m_orient3d(&P[3 * a], &P[3 * b], &P[3 * c], p);

                if (!((sw > 0 && sp > 0) || (sw < 0 && sp < 0)) || std::fabs(sp) < 1e-12) {
                    ok = false;    // p does not see this face -> non-star-shaped
                    break;
                }

                // orient (a,b,c,p) to orient3d>0 (mesh convention)
                if (sp < 0) {
                    std::swap(a, b);
                }

                cand.faces[cand.nf++] = { a, b, c };
                newq = std::fmin(newq, md4(a, b, c, p));

                if (cand.nf > 6) {
                    ok = false;    // safety
                    break;
                }
            }
        }

        if (!ok || newq <= oldq + kGain) {
            continue;
        }

        cand.p[0] = p[0];
        cand.p[1] = p[1];
        cand.p[2] = p[2];
        ins.push_back(cand);
        dead[s] = dead[F] = touched[s] = touched[F] = 1;
    }

    if (ins.empty()) {
        return 0;
    }

    for (const Ins& c : ins) {
        int pidx = (int)m.numPoints();
        m.points.push_back(c.p[0]);
        m.points.push_back(c.p[1]);
        m.points.push_back(c.p[2]);
        m.point_marker.push_back(0);
        m.point_orig.push_back(-1);   // a new (interior) node

        if (!m.point_failed.empty()) {
            m.point_failed.push_back(0);
            m.point_sig.push_back(0);
        }
        int lab = m.tet_label[c.s];

        for (int i = 0; i < c.nf; ++i) {
            m.tets.push_back(c.faces[i][0]);
            m.tets.push_back(c.faces[i][1]);
            m.tets.push_back(c.faces[i][2]);
            m.tets.push_back(pidx);
            m.tet_label.push_back(lab);

            for (int k = 0; k < 4; ++k) {
                m.tet_neigh.push_back(-1);
                m.tet_face_marker.push_back(0);
            }
        }
    }

    dead.resize(static_cast<size_t>(m.numTets()), 0);
    compact_dead_cpu(m, dead);
    recompute_face_markers(m);

    if (verbose) TN_FPRINTF(stderr, "[steiner] %zu sliver insertions -> %lld tets\n",
                                 ins.size(), (long long)m.numTets());

    return (int)ins.size();
}

}  // namespace

size_t optimize_mesh(TetOut& out, Nodes& nd, const OptParams& prm, OptStats& os) {
    OmpThreadCap cap;
    CoarseCDT m;
    const int64_t np = static_cast<int64_t>(out.P.size() / 3), nt = static_cast<int64_t>(out.label.size());
    m.points.assign(out.P.begin(), out.P.end());
    m.tets.assign(out.tets.begin(), out.tets.end());
    m.tet_label.assign(out.label.begin(), out.label.end());
    m.point_marker.resize(static_cast<size_t>(np));
    m.point_orig.resize(static_cast<size_t>(np));

    for (int64_t v = 0; v < np; ++v) {
        m.point_marker[v] = v < static_cast<int64_t>(nd.size()) && nd.typ[v] != TN_INTERIOR ? 1 : 0;
        m.point_orig[v] = static_cast<int>(v);
    }

    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < nt; ++t) {   // orient3d(a,b,c,d) > 0, the convention every pass creates tets in
        int* v = &m.tets[4 * t];

        if (b2m_orient3d(&m.points[3 * v[0]], &m.points[3 * v[1]], &m.points[3 * v[2]], &m.points[3 * v[3]]) < 0.0) {
            std::swap(v[2], v[3]);
        }
    }

    m.tet_face_marker.assign(static_cast<size_t>(nt) * 4, 0);
    compact_dead_cpu(m, std::vector<char>(static_cast<size_t>(nt), 0));   // adjacency
    recompute_face_markers(m);

    if (prm.verbose) {
        quality_report(m, "pre-opt");
    }

    g_opt_guard = prm.q > 0.0;
    g_opt_q = prm.q;
    const auto t0 = std::chrono::steady_clock::now();
    // label sets of the trussnet nodes ({a}, {a,b}, {a,b,c}, {a,b,c,d}); a node
    // created here (Steiner) is interior: its set is empty (never a kite corner)
    std::vector<std::array<int, 4>> lsets(nd.size());

    for (size_t v = 0; v < nd.size(); ++v) {
        lsets[v] = { { nd.lab[v], nd.typ[v] >= 1 ? nd.part[2 * v] : -1, nd.typ[v] >= 2 ? nd.part[2 * v + 1] : -1,
                       nd.typ[v] >= 3 && v < nd.part3.size() ? nd.part3[v] : -1 } };
    }

    auto lset = [&](int v) {
        const int s = v < static_cast<int>(m.point_orig.size()) ? m.point_orig[v] : -1;
        return s >= 0 ? lsets[s] : std::array<int, 4>{ { -1, -1, -1, -1 } };
    };

    typedef std::chrono::steady_clock oclk;
    auto lap = [](oclk::time_point& t) {
        const oclk::time_point n = oclk::now();
        const double d = std::chrono::duration<double, std::milli>(n - t).count();
        t = n;
        return d;
    };

    // trussnet: a pass that found nothing in the previous round is skipped while
    // the rest of the round leaves the topology alone, and the loop ends once a
    // round changes no topology and moves < 0.1% of the nodes (later rounds of
    // the gpu_brain2mesh loop moved a handful of vertices at full-scan cost)
    int last[6] = { 1, 1, 1, 1, 1, 1 };

    for (int round = 0; round < prm.max_rounds; ++round) {
        oclk::time_point tp = oclk::now();
        const bool topo_prev = round == 0 || last[0] + last[1] + last[2] + last[3] + last[4] > 0;
        const int nf = prm.flip32 && (last[0] || topo_prev) ? remove_slivers_32(m, 4, prm.verbose) : 0;
        os.ms_pass[0] += lap(tp);
        const int nk = prm.kites && (last[1] || nf) ? flatten_kites(m, prm.kite_deg, lset, prm.verbose) : 0;
        os.ms_pass[1] += lap(tp);
        os.kites += nk;
        const int n23 = prm.flip23 && (last[2] || (round > 0 && nf + nk > 0 && last[2])) ? flip_23(m, prm.verbose) : 0;
        os.ms_pass[2] += lap(tp);
        const int nc = prm.collapse && (last[3] || nf + nk + n23 > 0) ? collapse_interior(m, prm.verbose) : 0;
        os.ms_pass[3] += lap(tp);
        const int ns = prm.steiner && (last[4] || nf + nk + n23 + nc > 0) ? insert_steiner_slivers(m, prm.verbose) : 0;
        os.ms_pass[4] += lap(tp);
        const int nm = prm.smooth ? smooth_interior(m, 4, prm.verbose) : 0;
        os.ms_pass[5] += lap(tp);
        last[0] = nf;
        last[1] = nk;
        last[2] = n23;
        last[3] = nc;
        last[4] = ns;
        last[5] = nm;
        os.flips32 += nf;
        os.flips23 += n23;
        os.collapses += nc;
        os.steiner += ns;
        os.moves += nm;
        ++os.rounds;

        if (nf == 0 && nk == 0 && n23 == 0 && nc == 0 && ns == 0 && nm < std::max<int64_t>(1, m.numPoints() / 1000)) {
            break;
        }
    }

    os.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    g_opt_guard = false;
    g_opt_q = 0.0;

    if (prm.verbose) {
        quality_report(m, "post-opt");
    }

    // back to trussnet: points / tets / labels, and the node arrays remapped
    // through point_orig (new Steiner nodes are interior nodes of their tet's label)
    compact_points(m);
    const int64_t np2 = m.numPoints(), nt2 = m.numTets();
    std::vector<int> newlab(static_cast<size_t>(np2), -1);

    for (int64_t t = 0; t < nt2; ++t)
        for (int k = 0; k < 4; ++k) {
            newlab[m.tets[4 * t + k]] = m.tet_label[t];
        }

    Nodes o;
    o.P.resize(static_cast<size_t>(np2) * 3);
    o.lab.resize(np2);
    o.typ.resize(np2);
    o.part.resize(static_cast<size_t>(np2) * 2);
    o.part3.resize(np2);

    for (int64_t v = 0; v < np2; ++v) {
        const int s = m.point_orig[v];

        for (int k = 0; k < 3; ++k) {
            o.P[3 * v + k] = static_cast<float>(m.points[3 * v + k]);
        }

        if (s >= 0 && s < static_cast<int>(nd.size())) {
            o.lab[v] = nd.lab[s];
            o.typ[v] = nd.typ[s];
            o.part[2 * v] = nd.part[2 * s];
            o.part[2 * v + 1] = nd.part[2 * s + 1];
            o.part3[v] = s < static_cast<int>(nd.part3.size()) ? nd.part3[s] : TN_NOLAB;
        } else {
            o.lab[v] = static_cast<uint16_t>(std::max(0, newlab[v]));
            o.typ[v] = TN_INTERIOR;
            o.part[2 * v] = o.part[2 * v + 1] = o.part3[v] = TN_NOLAB;
        }
    }

    nd = std::move(o);
    out.P = nd.P;
    out.tets.assign(m.tets.begin(), m.tets.end());
    out.label.assign(m.tet_label.begin(), m.tet_label.end());
    return static_cast<size_t>(os.flips32 + os.kites + os.flips23 + os.collapses + os.steiner + os.moves);
}

}  // namespace tn
