// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_particles.cpp -- see v2m_particles.h. Host (OpenMP) driver of the seeding and
// particle bodies; the round structure matches the device path.

#include "v2m_particles.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <vector>

#include "v2m_log.h"
#include "v2m_omp.h"

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
#define V2M_G
#include "opencl/v2m_grid_body.cl"
#include "opencl/v2m_sdf_body.cl"
#include "opencl/v2m_seed_body.cl"
#include "opencl/v2m_particle_body.cl"
#undef V2M_G

}  // namespace particle_host

using namespace particle_host;

namespace {

V2mDims dims_of(const Grid& g) {
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
    return d;
}

typedef std::chrono::steady_clock clk;
double since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

}  // namespace

#define GRID_FIELD d, g.L->data(), g.bl_cnt.data(), g.bl_lab.data(), g.bl_slot.data(), g.phi.data(), g.gI, g.gTW.data(), g.gm

// Junction-line seeds (v2m_junction_vertex): candidates at the 3-label grid
// vertices (count -> scan -> fill), one kept per (level, cell, label triple) --
// the smallest vertex index, by a sort on the key (a radix sort / atomic-min
// claim on the device) -- then lattice nodes closer than 0.4 h to a kept seed are
// dropped (compaction) so no near-duplicates reach the Delaunay stage.
static void seed_junctions(const Grid& g, const RelaxParams& prm, Nodes& nd) {
    const V2mDims d = dims_of(g);
    const int vx1 = g.nx + 1, vy1 = g.ny + 1, vz1 = g.nz + 1;
    const int64_t nvert = static_cast<int64_t>(vx1) * vy1 * vz1;
    struct Cand {
        int key[7];   // level, cell x/y/z, sorted label triple
        int64_t vert;
        float x[3];
        int lab3[3];
    };
    std::vector<int> cc(static_cast<size_t>(nvert) + 1, 0);
    auto vijk = [&](int64_t v, int& i, int& j, int& k) {
        i = static_cast<int>(v % vx1) - 1;
        j = static_cast<int>((v / vx1) % vy1) - 1;
        k = static_cast<int>(v / (static_cast<int64_t>(vx1) * vy1)) - 1;
    };
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t v = 0; v < nvert; ++v) {
        int i, j, k, l3[3], key[4];
        float x[3];
        vijk(v, i, j, k);
        cc[v + 1] = v2m_junction_vertex(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.jseed, i, j, k, x,
                                       l3, key);
    }

    for (int64_t v = 0; v < nvert; ++v) {
        cc[v + 1] += cc[v];
    }

    std::vector<Cand> cand(cc[nvert]);
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t v = 0; v < nvert; ++v) {
        if (cc[v + 1] == cc[v]) {
            continue;
        }

        int i, j, k, key[4];
        Cand& c = cand[cc[v]];
        vijk(v, i, j, k);
        v2m_junction_vertex(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.jseed, i, j, k, c.x, c.lab3,
                           key);
        int t[3] = { c.lab3[0], c.lab3[1], c.lab3[2] };
        std::sort(t, t + 3);

        for (int e = 0; e < 4; ++e) {
            c.key[e] = key[e];
        }

        for (int e = 0; e < 3; ++e) {
            c.key[4 + e] = t[e];
        }

        c.vert = v;
    }

    std::sort(cand.begin(), cand.end(), [](const Cand& p, const Cand& q) {
        for (int e = 0; e < 7; ++e)
            if (p.key[e] != q.key[e]) {
                return p.key[e] < q.key[e];
            }

        return p.vert < q.vert;
    });
    std::vector<Cand> keep;

    for (size_t s = 0; s < cand.size(); ++s)
        if (s == 0 || !std::equal(cand[s].key, cand[s].key + 7, cand[s - 1].key)) {
            keep.push_back(cand[s]);
        }

    // drop lattice nodes within 0.4 h of a seed (uniform hash of the seeds)
    const float cell = 0.4f * g.hmax;
    const float ext[3] = { (g.nx + 1) * g.vs[0], (g.ny + 1) * g.vs[1], (g.nz + 1) * g.vs[2] };
    const int hd[3] = { static_cast<int>(ext[0] / cell) + 2, static_cast<int>(ext[1] / cell) + 2,
                        static_cast<int>(ext[2] / cell) + 2 };
    auto hkey = [&](const float* p, int* c3) {
        for (int e = 0; e < 3; ++e) {
            c3[e] = std::min(hd[e] - 1, std::max(0, static_cast<int>(std::floor((p[e] + g.vs[e]) / cell))));
        }

        return c3[0] + hd[0] * (c3[1] + static_cast<int64_t>(hd[1]) * c3[2]);
    };
    std::vector<int64_t> bkey(keep.size());
    std::vector<int> order(keep.size());

    for (size_t s = 0; s < keep.size(); ++s) {
        int c3[3];
        bkey[s] = hkey(keep[s].x, c3);
        order[s] = static_cast<int>(s);
    }

    std::sort(order.begin(), order.end(), [&](int p, int q) {
        return bkey[p] < bkey[q];
    });
    const int n = static_cast<int>(nd.size());
    std::vector<char> drop(n, 0);
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int i = 0; i < n; ++i) {
        if (nd.typ[i] == V2M_CORNER) {
            continue;
        }

        const float* p = &nd.P[3 * i];
        const float h = v2m_h_at(d, g.h.data(), p[0], p[1], p[2]);
        int c3[3];
        hkey(p, c3);

        for (int dz = -1; dz <= 1 && !drop[i]; ++dz)
            for (int dy = -1; dy <= 1 && !drop[i]; ++dy)
                for (int dx = -1; dx <= 1 && !drop[i]; ++dx) {
                    const int x = c3[0] + dx, y = c3[1] + dy, z = c3[2] + dz;

                    if (x < 0 || y < 0 || z < 0 || x >= hd[0] || y >= hd[1] || z >= hd[2]) {
                        continue;
                    }

                    const int64_t kk = x + hd[0] * (y + static_cast<int64_t>(hd[1]) * z);
                    auto lo = std::lower_bound(order.begin(), order.end(), kk, [&](int s, int64_t v) {
                        return bkey[s] < v;
                    });

                    for (auto it = lo; it != order.end() && bkey[*it] == kk; ++it) {
                        const float* q = keep[*it].x;
                        const float ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] - q[2];

                        if (ex * ex + ey * ey + ez * ez < 0.16f * h * h) {
                            drop[i] = 1;
                            break;
                        }
                    }
                }
    }

    Nodes out;
    out.P.reserve(nd.P.size() + 3 * keep.size());

    for (int i = 0; i < n; ++i)
        if (!drop[i]) {
            out.P.insert(out.P.end(), &nd.P[3 * i], &nd.P[3 * i] + 3);
            out.lab.push_back(nd.lab[i]);
            out.typ.push_back(nd.typ[i]);
            out.part.push_back(nd.part[2 * i]);
            out.part.push_back(nd.part[2 * i + 1]);
            out.part3.push_back(nd.part3[i]);
        }

    for (const Cand& c : keep) {
        out.P.insert(out.P.end(), c.x, c.x + 3);
        out.lab.push_back(static_cast<uint16_t>(c.lab3[0]));
        out.typ.push_back(V2M_JUNCTION);
        out.part.push_back(static_cast<uint16_t>(c.lab3[1]));
        out.part.push_back(static_cast<uint16_t>(c.lab3[2]));
        out.part3.push_back(V2M_NOLAB);
    }

    if (prm.verbose) {
        V2M_FPRINTF(stderr, "[seed]  junction lines: %zu candidates -> %zu seeds, %d lattice nodes dropped\n",
                   cand.size(), keep.size(), n - static_cast<int>(out.size()) + static_cast<int>(keep.size()));
    }

    nd = std::move(out);
}

// Reorder the nodes along a Morton (Z-order) curve of their voxel coordinates:
// the truss neighbours of a node are then close in memory (coherent gathers on
// the GPU, cache hits on the CPU) and the Delaunay inserts them spatially sorted.
static void morton_order(const Grid& g, Nodes& nd) {
    const int n = static_cast<int>(nd.size());
    auto spread = [](uint64_t x) {   // 21 bits -> every third bit
        x &= 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffffULL;
        x = (x | x << 16) & 0x1f0000ff0000ffULL;
        x = (x | x << 8) & 0x100f00f00f00f00fULL;
        x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
        x = (x | x << 2) & 0x1249249249249249ULL;
        return x;
    };
    std::vector<std::pair<uint64_t, int>> key(n);
    #pragma omp parallel for

    for (int i = 0; i < n; ++i) {
        const uint64_t x = static_cast<uint64_t>(std::max(0.0f, 2.0f * nd.P[3 * i] / g.vs[0] + 2.0f)),
                       y = static_cast<uint64_t>(std::max(0.0f, 2.0f * nd.P[3 * i + 1] / g.vs[1] + 2.0f)),
                       z = static_cast<uint64_t>(std::max(0.0f, 2.0f * nd.P[3 * i + 2] / g.vs[2] + 2.0f));
        key[i] = std::make_pair(spread(x) | spread(y) << 1 | spread(z) << 2, i);
    }

    std::sort(key.begin(), key.end());
    Nodes o;
    o.P.resize(nd.P.size());
    o.lab.resize(n);
    o.typ.resize(n);
    o.part.resize(nd.part.size());
    o.part3.resize(n);
    #pragma omp parallel for

    for (int k = 0; k < n; ++k) {
        const int i = key[k].second;

        for (int e = 0; e < 3; ++e) {
            o.P[3 * k + e] = nd.P[3 * i + e];
        }

        o.lab[k] = nd.lab[i];
        o.typ[k] = nd.typ[i];
        o.part[2 * k] = nd.part[2 * i];
        o.part[2 * k + 1] = nd.part[2 * i + 1];
        o.part3[k] = nd.part3[i];
    }

    nd = std::move(o);
}

static void seed_cpu_body(const Grid& g, const RelaxParams& prm, Nodes& nd);

// Shape input: pinned nodes on the primitives' sharp features (g.feat: corners,
// edge segments, rim circles), spaced by the sizing along each curve, kept where
// the labels round the point differ (a crease of the composed regions, not one
// hidden under a later shape). Each is a CORNER node (fixed through relaxation,
// tessellation and the optimiser) with the labels found round it; the seeds
// within 0.4 h of one are dropped.
static void add_feature_nodes(const Grid& g, Nodes& nd) {
    const V2mDims d = dims_of(g);
    std::vector<float> fp;          // candidates: x y z (corners, then the curves' label changes, then the rest)
    std::vector<float> tp, rp;      // (the curves' label changes; their regular points)
    auto hat = [&](const float* p) {
        return v2m_h_at(d, g.h.data(), p[0], p[1], p[2]);
    };
    // the labels round x (x and 6 samples at 0.2 h), as bits
    auto mask_at = [&](const float* x) {
        static const float dir[7][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        const float e = 0.2f * hat(x);
        uint64_t m = 0;

        for (int s = 0; s < 7; ++s) {
            int sec;
            float mg;
            const int l = v2m_label_of(GRID_FIELD, 0, x[0] + e * dir[s][0], x[1] + e * dir[s][1], x[2] + e * dir[s][2], &sec, &mg);
            m |= l < 64 ? uint64_t(1) << l : 0;
        }

        return m;
    };
    auto curve = [&](int type, const float* q) {   // points along a segment / circle, h apart
        const int nsamp = 256;
        std::vector<float> pts(3 * (nsamp + 1));
        float u[3] = { 0, 0, 0 }, w[3] = { 0, 0, 0 };

        if (type == 3) {   // an orthonormal frame about the circle's normal
            const float* n = q + 3;
            const float a0 = std::fabs(n[0]) < 0.9f ? 1.0f : 0.0f, a1 = a0 > 0 ? 0.0f : 1.0f;
            u[0] = n[1] * 0 - n[2] * a1;
            u[1] = n[2] * a0 - n[0] * 0;
            u[2] = n[0] * a1 - n[1] * a0;
            const float lu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

            for (float& x : u) {
                x /= lu;
            }

            w[0] = n[1] * u[2] - n[2] * u[1];
            w[1] = n[2] * u[0] - n[0] * u[2];
            w[2] = n[0] * u[1] - n[1] * u[0];
        }

        auto at = [&](float t, float* o) {   // the curve at parameter t in [0, 1]
            if (type == 4) {   // a polyline: [n, points], by its points' index
                const int n = static_cast<int>(q[0]);
                const float u = t * static_cast<float>(n - 1);
                const int i = std::min(n - 2, std::max(0, static_cast<int>(u)));
                const float f = u - static_cast<float>(i);

                for (int a = 0; a < 3; ++a) {
                    o[a] = q[1 + 3 * i + a] + f * (q[1 + 3 * (i + 1) + a] - q[1 + 3 * i + a]);
                }
            } else if (type == 2) {
                for (int a = 0; a < 3; ++a) {
                    o[a] = q[a] + t * (q[3 + a] - q[a]);
                }
            } else {
                const float th = 6.28318531f * t, c = std::cos(th), s = std::sin(th);

                for (int a = 0; a < 3; ++a) {
                    o[a] = q[a] + q[6] * (c * u[a] + s * w[a]);
                }
            }
        };

        for (int k = 0; k <= nsamp; ++k) {
            at(static_cast<float>(k) / nsamp, &pts[3 * k]);
        }

        // where the labels round the curve change (an edge crossing an interface:
        // a point of the edge and of the interface's junction with it), pinned first
        {
            std::vector<uint64_t> mk(nsamp + 1);

            for (int k = 0; k <= nsamp; ++k) {
                mk[static_cast<size_t>(k)] = mask_at(&pts[3 * k]);
            }

            for (int k = 1; k <= nsamp; ++k) {
                if (mk[static_cast<size_t>(k)] == mk[static_cast<size_t>(k - 1)]) {
                    continue;
                }

                float lo = static_cast<float>(k - 1) / nsamp, hi = static_cast<float>(k) / nsamp, x[3];

                for (int it = 0; it < 14; ++it) {
                    const float mid = 0.5f * (lo + hi);
                    at(mid, x);

                    if (mask_at(x) == mk[static_cast<size_t>(k - 1)]) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }

                at(0.5f * (lo + hi), x);
                tp.insert(tp.end(), x, x + 3);
            }
        }

        // arc length in units of h: n = round(total) segments (>= 1; a circle >= 6)
        std::vector<double> acc(nsamp + 1, 0.0);

        for (int k = 1; k <= nsamp; ++k) {
            const float* a = &pts[3 * (k - 1)], *b = &pts[3 * k];
            const float ds = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
            const float m[3] = { 0.5f * (a[0] + b[0]), 0.5f * (a[1] + b[1]), 0.5f * (a[2] + b[2]) };
            acc[k] = acc[k - 1] + ds / std::max(hat(m), 1e-6f);
        }

        // (a polyline -- two surfaces crossing, four regions round it -- twice as
        // dense: a tet near it must reach one of its nodes, or it spans two
        // surfaces that share no label)
        const double per = type == 4 ? 2.0 : 1.0;
        const int ns = std::max(type == 3 ? 6 : 1, static_cast<int>(std::lround(per * acc[nsamp])));

        // the interior points (a segment's ends are its corners; a circle, a polyline, from its start)
        for (int s = type == 3 || type == 4 ? 0 : 1; s < ns; ++s) {
            const double target = acc[nsamp] * s / ns;
            int k = 1;

            while (k < nsamp && acc[k] < target) {
                ++k;
            }

            const double f = acc[k] > acc[k - 1] ? (target - acc[k - 1]) / (acc[k] - acc[k - 1]) : 0.0;

            for (int a = 0; a < 3; ++a) {
                rp.push_back(static_cast<float>(pts[3 * (k - 1) + a] + f * (pts[3 * k + a] - pts[3 * (k - 1) + a])));
            }
        }
    };

    for (size_t k = 0; k < g.feat.size();) {
        const int type = static_cast<int>(g.feat[k]);
        const float* q = &g.feat[k + 1];

        if (type == 1) {
            fp.insert(fp.end(), { q[0], q[1], q[2] });
        } else if (type != 4 || q[0] >= 2) {
            curve(type, q);
        }

        k += type == 1 ? 4 : type == 2 ? 7 : type == 4 ? 2 + 3 * static_cast<size_t>(q[0]) : 8;
    }

    fp.insert(fp.end(), tp.begin(), tp.end());
    fp.insert(fp.end(), rp.begin(), rp.end());

    // each candidate: the labels round it (samples at 0.25 h, kept if >= 2): the 14
    // axis / diagonal directions and 128 spread over the sphere (a golden spiral,
    // ~16 degrees apart) -- a knife edge's thin wedge (a box less a larger sphere:
    // 31 degrees at its holes' rims) falls between the 14 alone, and its pins went
    std::vector<std::array<float, 3>> dirs = {
        { { 1, 0, 0 } }, { { -1, 0, 0 } }, { { 0, 1, 0 } }, { { 0, -1, 0 } }, { { 0, 0, 1 } }, { { 0, 0, -1 } },
        { { .577f, .577f, .577f } }, { { -.577f, .577f, .577f } }, { { .577f, -.577f, .577f } }, { { .577f, .577f, -.577f } },
        { { -.577f, -.577f, .577f } }, { { -.577f, .577f, -.577f } }, { { .577f, -.577f, -.577f } }, { { -.577f, -.577f, -.577f } }
    };

    for (int i = 0; i < 128; ++i) {
        const double zc = 1 - (2 * i + 1) / 128.0, rr = std::sqrt(std::max(0.0, 1 - zc * zc)), ph = 2.39996322972865 * i;
        dirs.push_back({ { static_cast<float>(rr * std::cos(ph)), static_cast<float>(rr * std::sin(ph)), static_cast<float>(zc) } });
    }

    const size_t nc = fp.size() / 3;
    std::vector<std::array<int, 4>> ls(nc);
    std::vector<int> nls(nc, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t c = 0; c < static_cast<int64_t>(nc); ++c) {
        const float* x = &fp[3 * static_cast<size_t>(c)];

        if (x[0] < 0 || x[1] < 0 || x[2] < 0 || x[0] > (g.nx - 1) * g.vs[0] || x[1] > (g.ny - 1) * g.vs[1] || x[2] > (g.nz - 1) * g.vs[2]) {
            continue;
        }

        const float e = 0.25f * hat(x);
        std::map<int, int> cnt;

        for (const auto& dv : dirs) {
            int sec;
            float mg;
            const int l = v2m_label_of(GRID_FIELD, 0, x[0] + e * dv[0], x[1] + e * dv[1], x[2] + e * dv[2], &sec, &mg);
            ++cnt[l];
        }

        if (cnt.size() < 2) {
            continue;   // hidden: inside one region
        }

        std::vector<std::pair<int, int>> by(cnt.begin(), cnt.end());
        std::sort(by.begin(), by.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        // (the node's own label a tissue one: 0, the exterior, only as a partner)
        std::stable_partition(by.begin(), by.end(), [](const std::pair<int, int>& a) {
            return a.first != 0;
        });
        int m = 0;

        for (const auto& pr : by)
            if (m < 4) {
                ls[static_cast<size_t>(c)][static_cast<size_t>(m++)] = pr.first;
            }

        if (m > 0 && ls[static_cast<size_t>(c)][0] == 0) {
            m = 0;   // (no tissue round it)
        }

        nls[static_cast<size_t>(c)] = m;
    }

    // keep: one per 0.3 h (corners first: they came first), then drop the seeds near them
    std::vector<float> keepP;
    std::vector<std::array<int, 4>> keepL;
    std::vector<int> keepN;

    for (size_t c = 0; c < nc; ++c) {
        if (nls[c] < 2) {
            continue;
        }

        const float* x = &fp[3 * c];
        const float r = 0.3f * hat(x);
        bool near = false;

        for (size_t k = 0; k < keepP.size() / 3 && !near; ++k) {
            const float dx = keepP[3 * k] - x[0], dy = keepP[3 * k + 1] - x[1], dz = keepP[3 * k + 2] - x[2];
            near = dx * dx + dy * dy + dz * dz < r * r;
        }

        if (!near) {
            keepP.insert(keepP.end(), x, x + 3);
            keepL.push_back(ls[c]);
            keepN.push_back(nls[c]);
        }
    }

    const size_t nk = keepP.size() / 3;

    if (nk == 0) {
        return;
    }

    // the seeds within 0.4 h of a feature node leave (a hash of the feature nodes)
    const float cell = 0.4f * g.hmax > 0 ? 0.4f * g.hmax : 1.0f;
    auto key = [&](int64_t i, int64_t j, int64_t k) {
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };
    std::unordered_map<int64_t, std::vector<int>> hash;

    for (size_t k = 0; k < nk; ++k) {
        hash[key(static_cast<int64_t>(std::floor(keepP[3 * k] / cell)), static_cast<int64_t>(std::floor(keepP[3 * k + 1] / cell)),
                 static_cast<int64_t>(std::floor(keepP[3 * k + 2] / cell)))].push_back(static_cast<int>(k));
    }

    const size_t n0 = nd.size();
    std::vector<char> drop(n0, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(n0); ++i) {
        const float* x = &nd.P[3 * static_cast<size_t>(i)];
        const float r = 0.4f * hat(x);
        const int64_t ci = static_cast<int64_t>(std::floor(x[0] / cell)), cj = static_cast<int64_t>(std::floor(x[1] / cell)),
                      ck = static_cast<int64_t>(std::floor(x[2] / cell));
        const int64_t reach = static_cast<int64_t>(std::ceil(r / cell));

        for (int64_t a = -reach; a <= reach && !drop[static_cast<size_t>(i)]; ++a)
            for (int64_t b = -reach; b <= reach && !drop[static_cast<size_t>(i)]; ++b)
                for (int64_t c = -reach; c <= reach && !drop[static_cast<size_t>(i)]; ++c) {
                    auto it = hash.find(key(ci + a, cj + b, ck + c));

                    if (it == hash.end()) {
                        continue;
                    }

                    for (int k : it->second) {
                        const float dx = keepP[3 * static_cast<size_t>(k)] - x[0], dy = keepP[3 * static_cast<size_t>(k) + 1] - x[1],
                                    dz = keepP[3 * static_cast<size_t>(k) + 2] - x[2];

                        if (dx * dx + dy * dy + dz * dz < r * r) {
                            drop[static_cast<size_t>(i)] = 1;
                            break;
                        }
                    }
                }
    }

    Nodes out;
    const bool p3 = nd.part3.size() == n0;

    for (size_t i = 0; i < n0; ++i) {
        if (drop[i]) {
            continue;
        }

        out.P.insert(out.P.end(), nd.P.begin() + 3 * static_cast<std::ptrdiff_t>(i), nd.P.begin() + 3 * static_cast<std::ptrdiff_t>(i) + 3);
        out.lab.push_back(nd.lab[i]);
        out.typ.push_back(nd.typ[i]);
        out.part.push_back(nd.part[2 * i]);
        out.part.push_back(nd.part[2 * i + 1]);
        out.part3.push_back(p3 ? nd.part3[i] : static_cast<uint16_t>(V2M_NOLAB));
    }

    for (size_t k = 0; k < nk; ++k) {
        out.P.insert(out.P.end(), &keepP[3 * k], &keepP[3 * k] + 3);
        const std::array<int, 4>& L = keepL[k];
        const int m = keepN[k];
        out.lab.push_back(static_cast<uint16_t>(L[0]));
        out.typ.push_back(V2M_CORNER);
        out.part.push_back(static_cast<uint16_t>(m > 1 ? L[1] : V2M_NOLAB));
        out.part.push_back(static_cast<uint16_t>(m > 2 ? L[2] : V2M_NOLAB));
        out.part3.push_back(static_cast<uint16_t>(m > 3 ? L[3] : V2M_NOLAB));
    }

    nd = std::move(out);
}

void seed_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd) {
    seed_cpu_body(g, prm, nd);

    if (!g.feat.empty()) {
        add_feature_nodes(g, nd);
    }

    if (std::getenv("V2M_MORTON")) {   // (measured slower on the GPU move: kept as an option)
        morton_order(g, nd);
    }
}

static void seed_cpu_body(const Grid& g, const RelaxParams& prm, Nodes& nd) {
    OmpThreadCap cap;
    const V2mDims d = dims_of(g);
    const int64_t nv = static_cast<int64_t>(g.nx) * g.ny * g.nz;
    std::vector<int> cnt(static_cast<size_t>(nv) + 1, 0);
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int64_t v = 0; v < nv; ++v) {
        const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
        cnt[v + 1] = v2m_seed_voxel(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.voxel_trap ? 1 : 0, i,
                                   j, k, 0, nullptr, nullptr, 0);
    }

    for (int64_t v = 0; v < nv; ++v) {
        cnt[v + 1] += cnt[v];
    }

    const int n = cnt[nv];
    nd.P.assign(static_cast<size_t>(n) * 3, 0.0f);
    nd.lab.assign(n, 0);
    nd.typ.assign(n, V2M_INTERIOR);
    nd.part.assign(static_cast<size_t>(n) * 2, V2M_NOLAB);
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int64_t v = 0; v < nv; ++v) {
        if (cnt[v + 1] == cnt[v]) {
            continue;
        }

        const int i = static_cast<int>(v % g.nx), j = static_cast<int>((v / g.nx) % g.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(g.nx) * g.ny));
        v2m_seed_voxel(GRID_FIELD, g.grade.data(), prm.nseed, g.hmin, g.hmax, prm.voxel_trap ? 1 : 0, i, j, k, 1,
                      nd.P.data(), nd.lab.data(), cnt[v]);
    }

    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int i = 0; i < n; ++i) {
        v2m_seed_classify(GRID_FIELD, g.h.data(), i, nd.P.data(), nd.lab.data(), nd.typ.data(), nd.part.data());
    }

    nd.part3.assign(n, V2M_NOLAB);

    if (prm.jseed > 0.0f) {
        seed_junctions(g, prm, nd);
    }

    if (!prm.corners) {
        return;
    }

    const int n0 = static_cast<int>(nd.size());   // after the junction pass

    // fixed CORNER nodes at the grid vertices where >= 4 labels meet (count ->
    // scan -> fill over the (nx+1)(ny+1)(nz+1) vertices), appended after the lattice
    const int vx1 = g.nx + 1, vy1 = g.ny + 1, vz1 = g.nz + 1;
    const int64_t nvert = static_cast<int64_t>(vx1) * vy1 * vz1;
    std::vector<int> cc(static_cast<size_t>(nvert) + 1, 0);
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t v = 0; v < nvert; ++v) {
        const int i = static_cast<int>(v % vx1) - 1, j = static_cast<int>((v / vx1) % vy1) - 1,
                  k = static_cast<int>(v / (static_cast<int64_t>(vx1) * vy1)) - 1;
        cc[v + 1] = v2m_corner_vertex(GRID_FIELD, i, j, k, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    }

    for (int64_t v = 0; v < nvert; ++v) {
        cc[v + 1] += cc[v];
    }

    const int nc = cc[nvert];
    nd.P.resize(static_cast<size_t>(n0 + nc) * 3);
    nd.lab.resize(n0 + nc);
    nd.typ.resize(n0 + nc);
    nd.part.resize(static_cast<size_t>(n0 + nc) * 2);
    nd.part3.resize(n0 + nc, V2M_NOLAB);
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t v = 0; v < nvert; ++v) {
        if (cc[v + 1] == cc[v]) {
            continue;
        }

        const int i = static_cast<int>(v % vx1) - 1, j = static_cast<int>((v / vx1) % vy1) - 1,
                  k = static_cast<int>(v / (static_cast<int64_t>(vx1) * vy1)) - 1;
        v2m_corner_vertex(GRID_FIELD, i, j, k, 1, nd.P.data(), nd.lab.data(), nd.typ.data(), nd.part.data(),
                         nd.part3.data(), n0 + cc[v]);
    }
}

void relax_cpu(const Grid& g, const RelaxParams& prm, Nodes& nd, RelaxStats& st) {
    // Verlet trigger: rebuild once more than 1/rebuild_div of the nodes moved skin/2
    // since the build (nodes past a full skin refresh their own list anyway)
    static const int rebuild_div = std::getenv("V2M_REBUILD_DIV") ? std::atoi(std::getenv("V2M_REBUILD_DIV")) : 300;
    OmpThreadCap cap;
    const V2mDims d = dims_of(g);
    const int n = static_cast<int>(nd.size());
    const unsigned long long sdf0 = v2m_sdf_calls()[0], sdfg0 = v2m_sdf_calls()[1];   // (V2M_SDF_COUNT)

    // hash geometry: level-0 bin = the finest search radius
    V2mHash H;
    H.b0 = (prm.t + prm.skin) * g.hmin;
    H.ox = -0.5f * g.vs[0];
    H.oy = -0.5f * g.vs[1];
    H.oz = -0.5f * g.vs[2];
    const float ext[3] = { g.nx * g.vs[0], g.ny * g.vs[1], g.nz * g.vs[2] };
    H.nlev = 1;

    while (H.nlev < V2M_MAXLEV && H.b0 * static_cast<float>(1 << (H.nlev - 1)) < (prm.t + prm.skin) * g.hmax) {
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

    std::vector<int> key(n), cstart(static_cast<size_t>(nkeys) + 1), sorted(n), nbr(static_cast<size_t>(n) * V2M_K),
        nnb(n);
    std::vector<float> F(static_cast<size_t>(n) * 4), P0(nd.P), mv(n), hn(n), Ps;

    for (int i = 0; i < n; ++i) {   // h at each node (v2m_move keeps it current)
        hn[i] = v2m_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
    }

    auto rebuild = [&]() {
        clk::time_point t0 = clk::now();
        #pragma omp parallel for

        for (int i = 0; i < n; ++i) {
            const float h = v2m_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
            key[i] = v2m_bin_key(&H, v2m_level_of(&H, (prm.t + prm.skin) * h), nd.P[3 * i], nd.P[3 * i + 1],
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

        Ps.resize(static_cast<size_t>(n) * 4);
        #pragma omp parallel for schedule(static)

        for (int s2 = 0; s2 < n; ++s2) {   // positions + sizes in bin order (see v2m_neighbors)
            const int j = sorted[s2];
            Ps[4 * s2] = nd.P[3 * j];
            Ps[4 * s2 + 1] = nd.P[3 * j + 1];
            Ps[4 * s2 + 2] = nd.P[3 * j + 2];
            Ps[4 * s2 + 3] = hn[j];
        }

        #pragma omp parallel for schedule(monotonic: dynamic, 256)

        for (int i = 0; i < n; ++i) {
            float kds[V2M_K];
            int kids[V2M_K];
            v2m_neighbors(&H, hn.data(), nd.P.data(), nd.lab.data(), nd.typ.data(), cstart.data(), sorted.data(), Ps.data(),
                         prm.t, prm.skin, i, nbr.data(), nnb.data(), kds, kids, 1);
        }

        P0 = nd.P;
        ++st.rebuilds;
        st.ms_hash += since(t0);
    };

    rebuild();
    // FIRE: velocities and per-node power (see FireCtl)
    const int fire = prm.fire && !prm.voxel_trap ? 1 : 0;   // voxel trapping: Jacobi (see v2m_pipeline.cpp)
    std::vector<float> V(fire ? static_cast<size_t>(n) * 3 : 1, 0.0f), pw(fire ? n : 1, 0.0f);
    FireCtl fc(prm);

    for (int it = 0; it < prm.max_iters; ++it) {
        clk::time_point t0 = clk::now();
        #pragma omp parallel for schedule(monotonic: dynamic, 1024)

        for (int i = 0; i < n; ++i) {
            v2m_force(hn.data(), nd.P.data(), nd.typ.data(), nbr.data(), nnb.data(), prm.fscale, prm.fsurf, i,
                     &F[4 * i]);
        }

        st.ms_force += since(t0);
        clk::time_point t1 = clk::now();
        float mmax = 0.0f;
        #pragma omp parallel for schedule(monotonic: dynamic, 1024) reduction(max : mmax)

        for (int i = 0; i < n; ++i) {
            mv[i] = v2m_move(GRID_FIELD, g.h.data(), F.data(), prm.dt, prm.maxstep, prm.snap,
                            prm.voxel_trap ? 1 : 0, i, nd.P.data(), nd.lab.data(),
                            nd.typ.data(), nd.part.data(), hn.data(), fire, V.data(), fc.dt, fc.alpha, pw.data());
            mmax = std::max(mmax, mv[i]);
        }

        if (fire) {
            double pt = 0.0;   // serial: a fixed summation order keeps the run deterministic

            for (int i = 0; i < n; ++i) {
                pt += pw[i];
            }

            if (fc.update(pt)) {
                std::fill(V.begin(), V.end(), 0.0f);
            }
        }

        st.ms_move += since(t1);
        st.iters = it + 1;

        if (std::getenv("V2M_MOVE_DEBUG") && it >= prm.max_iters - 6) {
            const int w = static_cast<int>(std::max_element(mv.begin(), mv.end()) - mv.begin());
            V2M_FPRINTF(stderr, "[mvdbg] it %d node %d typ %d lab %d part %d %d mv %.3f P %.3f %.3f %.3f\n", it, w,
                       nd.typ[w], nd.lab[w], nd.part[2 * w], nd.part[2 * w + 1], mv[w], nd.P[3 * w], nd.P[3 * w + 1],
                       nd.P[3 * w + 2]);
        }
        st.last_move = mmax;
        st.fire_resets = fc.resets;
        st.fire_dt = fire ? fc.dt : 0.0f;

        // convergence on the 99th percentile of |dp|/h from a log2 histogram (a
        // reduction, not a sort): a handful of restless nodes (e.g. at thin
        // junctions) must not keep the whole mesh iterating
        int hist[32] = { 0 };
        #pragma omp parallel for reduction(+ : hist[:32])

        for (int i = 0; i < n; ++i) {
            const int b = mv[i] > 0.0f ? std::min(31, std::max(0, 20 + static_cast<int>(std::floor(std::log2(mv[i]))))) : 0;
            ++hist[b];
        }

        int acc = 0, b99 = 31;

        for (int b = 0; b < 32; ++b) {
            acc += hist[b];

            if (acc >= 0.99 * n) {
                b99 = b;
                break;
            }
        }

        const float p99 = std::ldexp(1.0f, b99 - 20 + 1);   // upper edge of the bin
        st.last_p99 = p99;

        if (prm.verbose && (it % 50 == 0)) {
            if (fire) {
                V2M_FPRINTF(stderr, "[relax] iter %d: max move %.4g h, p99 < %.3g h; FIRE dt %.3g, %d resets\n", it, mmax,
                           p99, fc.dt, fc.resets);
            } else {
                V2M_FPRINTF(stderr, "[relax] iter %d: max move %.4g h, p99 < %.3g h\n", it, mmax, p99);
            }
        }

        if (p99 < prm.dptol && (!fire || fc.may_stop())) {
            break;
        }

        // Verlet criterion: rebuild once more than 0.1% of the nodes have moved skin/2
        // since the build. A node that jumped a full skin (an interior node snapping
        // onto the surface, up to snap*h) only refreshes its own list against the
        // current bins; the others see it again at the next rebuild. Rebuilding for
        // every such one-off event cost a full rebuild per iteration.
        int nhalf = 0;
        #pragma omp parallel for reduction(+ : nhalf)

        for (int i = 0; i < n; ++i) {
            const float ex = nd.P[3 * i] - P0[3 * i], ey = nd.P[3 * i + 1] - P0[3 * i + 1],
                        ez = nd.P[3 * i + 2] - P0[3 * i + 2];
            const float h = v2m_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
            const float m2 = ex * ex + ey * ey + ez * ez, s2 = prm.skin * prm.skin * h * h;

            if (m2 > s2) {
                float kds[V2M_K];
                int kids[V2M_K];
                v2m_neighbors(&H, hn.data(), nd.P.data(), nd.lab.data(), nd.typ.data(), cstart.data(),
                             sorted.data(), Ps.data(), prm.t, prm.skin, i, nbr.data(), nnb.data(), kds, kids, 1);
                P0[3 * i] = nd.P[3 * i];
                P0[3 * i + 1] = nd.P[3 * i + 1];
                P0[3 * i + 2] = nd.P[3 * i + 2];
            } else {
                nhalf += m2 > 0.25f * s2;
            }
        }

        // loose trigger for the bulk of the relaxation, strict (0.1%) near the end so
        // the final positions are relaxed with exact lists (the loose one alone cost
        // a few conforming faces on the wedge / T-junction phantoms)
        const bool endgame = it >= prm.max_iters * 4 / 5 || p99 < 4.0f * prm.dptol;

        if (nhalf > n / (endgame ? std::max(rebuild_div, 1000) : rebuild_div)) {
            rebuild();
        }
    }

    if (std::getenv("V2M_MOVE_DEBUG")) {   // who is still moving
        int cnt[4] = { 0, 0, 0, 0 }, tot[4] = { 0, 0, 0, 0 };

        for (int i = 0; i < n; ++i) {
            tot[nd.typ[i] & 3]++;
            cnt[nd.typ[i] & 3] += mv[i] > 0.02f;
        }

        V2M_FPRINTF(stderr, "[relax] moving > 0.02 h at the end: interior %d/%d interface %d/%d junction %d/%d\n",
                   cnt[0], tot[0], cnt[1], tot[1], cnt[2], tot[2]);
    }

    if (std::getenv("V2M_SDF_COUNT") && g.gm < 0) {   // shape fields: evaluations per node move
        const double moves = static_cast<double>(st.iters) * n;   // (exact single-threaded: the counter is not atomic)
        V2M_FPRINTF(stderr, "[sdf]   relaxation: %llu field evaluations (%llu with a gradient) = %.1f (%.1f) per node move\n",
                   v2m_sdf_calls()[0] - sdf0, v2m_sdf_calls()[1] - sdfg0, (v2m_sdf_calls()[0] - sdf0) / moves,
                   (v2m_sdf_calls()[1] - sdfg0) / moves);
    }

    st.n_interior = st.n_interface = st.n_junction = st.n_corner = 0;

    for (uint8_t t : nd.typ) {
        st.n_interior += t == V2M_INTERIOR;
        st.n_interface += t == V2M_INTERFACE;
        st.n_junction += t == V2M_JUNCTION;
        st.n_corner += t == V2M_CORNER;
    }
}

// ---- node thinning (--thin B) ------------------------------------------------------
//
// Applied to the seeds, before the relaxation. The relaxation only repels and never
// removes a node, so the seeded node count is final: in a thin layer next to a much
// finer interface (the skull by the CSF, with a steep --grad) the graded lattices and
// the interface projection leave more nodes than the local h asks for. A greedy
// Poisson-disk pass removes every node that has a kept node closer than B h(i):
//   - an interface node competes with the nodes of its own interface (its label pair,
//     junction / corner nodes on it included), an interior node with every node of
//     its label; junction and corner nodes are always kept;
//   - interface nodes go first, then interior ones, finest h first (ties by index):
//     fine regions keep their nodes, coarse ones are thinned around them, and the
//     result is deterministic.
// Lattice / relaxed spacings are ~0.85-1.2 h, so B ~ 0.7 only touches crowded regions.
// Lookups: a multi-level grid (cells B hmin 2^L), each query scanning the 27 cells of
// the level whose cells cover its radius.
// A relaxed node that drifts in between two pinned nodes of a feature curve
// (onto a box edge: the blended fields round the crease draw the interface
// nodes to it) takes the edge's place -- the Delaunay joins both pins to it and
// the crease kinks. With no node inside the pair's diametral ball the segment
// is a Gabriel edge, which every Delaunay tessellation holds: so the nodes
// inside the balls (other than the pinned ones) are removed. A pair farther
// apart than 1.6 h is not a crease (the curve hidden under a later shape
// between them) and is left alone.
size_t protect_features(const Grid& g, Nodes& nd) {
    const size_t n = nd.size();

    if (g.feat.empty() || n == 0) {
        return 0;
    }

    const V2mDims d = dims_of(g);
    auto hat = [&](const float* p) {
        return v2m_h_at(d, g.h.data(), p[0], p[1], p[2]);
    };
    std::vector<int> pins;

    for (size_t i = 0; i < n; ++i)
        if (nd.typ[i] == V2M_CORNER) {
            pins.push_back(static_cast<int>(i));
        }

    std::vector<std::array<float, 4>> balls;   // centre, radius^2
    // (distance, parameter) of point p from feature curve `type` q
    auto locate = [&](int type, const float* q, const float* p, float& dist, float& t) {
        auto seg = [&](const float* a, const float* b, float& dd, float& tt) {
            float ab[3], ap[3], ab2 = 0, apab = 0;

            for (int k = 0; k < 3; ++k) {
                ab[k] = b[k] - a[k];
                ap[k] = p[k] - a[k];
                ab2 += ab[k] * ab[k];
                apab += ap[k] * ab[k];
            }

            tt = ab2 > 0 ? std::min(1.0f, std::max(0.0f, apab / ab2)) : 0.0f;
            dd = 0;

            for (int k = 0; k < 3; ++k) {
                const float e = ap[k] - tt * ab[k];
                dd += e * e;
            }

            dd = std::sqrt(dd);
        };

        if (type == 2) {
            seg(q, q + 3, dist, t);
        } else if (type == 4) {   // [n, points]: t = segment index + fraction
            const int m = static_cast<int>(q[0]);
            dist = 1e30f;

            for (int i = 0; i + 1 < m; ++i) {
                float dd, tt;
                seg(q + 1 + 3 * i, q + 4 + 3 * i, dd, tt);

                if (dd < dist) {
                    dist = dd;
                    t = static_cast<float>(i) + tt;
                }
            }
        } else {   // circle c n r: t = the angle in a frame about n
            const float* c = q, *nn = q + 3;
            float v[3], h = 0;

            for (int k = 0; k < 3; ++k) {
                v[k] = p[k] - c[k];
                h += v[k] * nn[k];
            }

            float rad[3], rho = 0;

            for (int k = 0; k < 3; ++k) {
                rad[k] = v[k] - h * nn[k];
                rho += rad[k] * rad[k];
            }

            rho = std::sqrt(rho);
            dist = std::hypot(h, rho - q[6]);
            const float a0 = std::fabs(nn[0]) < 0.9f ? 1.0f : 0.0f, a1 = a0 > 0 ? 0.0f : 1.0f;
            float u[3] = { -nn[2] * a1, nn[2] * a0, nn[0] * a1 - nn[1] * a0 };
            const float lu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

            for (float& x : u) {
                x /= lu;
            }

            const float w[3] = { nn[1] * u[2] - nn[2] * u[1], nn[2] * u[0] - nn[0] * u[2], nn[0] * u[1] - nn[1] * u[0] };
            t = std::atan2(rad[0] * w[0] + rad[1] * w[1] + rad[2] * w[2], rad[0] * u[0] + rad[1] * u[1] + rad[2] * u[2]);
        }
    };

    for (size_t k = 0; k < g.feat.size();) {
        const int type = static_cast<int>(g.feat[k]);
        const float* q = &g.feat[k + 1];
        k += type == 1 ? 4 : type == 2 ? 7 : type == 4 ? 2 + 3 * static_cast<size_t>(q[0]) : 8;

        if (type == 1 || (type == 4 && q[0] < 2)) {
            continue;
        }

        std::vector<std::pair<float, int>> on;   // (parameter, pin) of the pins on the curve

        for (int j : pins) {
            const float* p = &nd.P[3 * static_cast<size_t>(j)];
            float dist, t = 0;
            locate(type, q, p, dist, t);

            if (dist < 1e-3f * hat(p)) {
                on.emplace_back(t, j);
            }
        }

        std::sort(on.begin(), on.end());
        const size_t m = on.size();

        for (size_t i = 0; i + 1 < m + (type == 3 && m > 2 ? 1 : 0); ++i) {   // (a circle closes)
            const float* a = &nd.P[3 * static_cast<size_t>(on[i].second)], *b = &nd.P[3 * static_cast<size_t>(on[(i + 1) % m].second)];
            const float c[3] = { 0.5f * (a[0] + b[0]), 0.5f * (a[1] + b[1]), 0.5f * (a[2] + b[2]) };
            const float r2 = 0.25f * ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
            const float h = hat(c);

            if (r2 > 0 && r2 <= 0.64f * h * h) {
                balls.push_back({ { c[0], c[1], c[2], r2 } });
            }
        }
    }

    if (balls.empty()) {
        return 0;
    }

    // the balls, hashed by cells of the largest diameter
    float cs = 0;

    for (const auto& bl : balls) {
        cs = std::max(cs, 2.0f * std::sqrt(bl[3]));
    }

    auto key = [&](int64_t ix, int64_t iy, int64_t iz) {
        return static_cast<uint64_t>(ix + (1 << 20)) << 42 | static_cast<uint64_t>(iy + (1 << 20)) << 21 |
               static_cast<uint64_t>(iz + (1 << 20));
    };
    std::unordered_map<uint64_t, std::vector<int>> cells;

    for (size_t bi = 0; bi < balls.size(); ++bi) {
        const auto& bl = balls[bi];
        const float r = std::sqrt(bl[3]);
        int64_t lo[3], hi[3];

        for (int k = 0; k < 3; ++k) {
            lo[k] = static_cast<int64_t>(std::floor((bl[k] - r) / cs));
            hi[k] = static_cast<int64_t>(std::floor((bl[k] + r) / cs));
        }

        for (int64_t x = lo[0]; x <= hi[0]; ++x)
            for (int64_t y = lo[1]; y <= hi[1]; ++y)
                for (int64_t z = lo[2]; z <= hi[2]; ++z) {
                    cells[key(x, y, z)].push_back(static_cast<int>(bi));
                }
    }

    std::vector<char> drop(n, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        if (nd.typ[static_cast<size_t>(i)] == V2M_CORNER) {
            continue;
        }

        const float* p = &nd.P[3 * static_cast<size_t>(i)];
        const auto it = cells.find(key(static_cast<int64_t>(std::floor(p[0] / cs)), static_cast<int64_t>(std::floor(p[1] / cs)),
                                       static_cast<int64_t>(std::floor(p[2] / cs))));

        if (it == cells.end()) {
            continue;
        }

        for (int bi : it->second) {
            const auto& bl = balls[static_cast<size_t>(bi)];
            const float dx = p[0] - bl[0], dy = p[1] - bl[1], dz = p[2] - bl[2];

            if (dx * dx + dy * dy + dz * dz < bl[3]) {
                drop[static_cast<size_t>(i)] = 1;
                break;
            }
        }
    }

    const bool p3 = nd.part3.size() == n;
    size_t w = 0;

    for (size_t i = 0; i < n; ++i) {
        if (drop[i]) {
            continue;
        }

        for (int k = 0; k < 3; ++k) {
            nd.P[3 * w + k] = nd.P[3 * i + k];
        }

        nd.lab[w] = nd.lab[i];
        nd.typ[w] = nd.typ[i];
        nd.part[2 * w] = nd.part[2 * i];
        nd.part[2 * w + 1] = nd.part[2 * i + 1];

        if (p3) {
            nd.part3[w] = nd.part3[i];
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

    return n - w;
}

size_t thin_nodes(const Grid& g, const RelaxParams& prm, Nodes& nd) {
    const int n = static_cast<int>(nd.size());
    const float B = prm.thin;

    if (B <= 0.0f || n == 0) {
        return 0;
    }

    const V2mDims d = dims_of(g);
    std::vector<float> hn(n);
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < n; ++i) {
        hn[i] = v2m_h_at(d, g.h.data(), nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2]);
    }

    const float c0 = std::max(1e-6f, B * g.hmin);
    int nlev = 1;

    while (nlev < 12 && c0 * static_cast<float>(1 << (nlev - 1)) < B * g.hmax) {
        ++nlev;
    }

    auto cell = [&](int L, float x, float y, float z, int dx, int dy, int dz) {
        const float c = c0 * static_cast<float>(1 << L);
        const uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(std::floor(x / c)) + dx + (1 << 19)) & 0xfffff,
                       iy = static_cast<uint64_t>(static_cast<int64_t>(std::floor(y / c)) + dy + (1 << 19)) & 0xfffff,
                       iz = static_cast<uint64_t>(static_cast<int64_t>(std::floor(z / c)) + dz + (1 << 19)) & 0xfffff;
        return static_cast<uint64_t>(L) << 60 | ix << 40 | iy << 20 | iz;
    };
    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(static_cast<size_t>(n) * 2);
    std::vector<char> keep(n, 0);
    auto insert = [&](int j) {
        keep[j] = 1;

        for (int L = 0; L < nlev; ++L) {
            grid[cell(L, nd.P[3 * j], nd.P[3 * j + 1], nd.P[3 * j + 2], 0, 0, 0)].push_back(j);
        }
    };
    auto has = [&](int j, int l) {   // label l among node j's labels
        return nd.lab[j] == l || (nd.typ[j] != V2M_INTERIOR && nd.part[2 * j] == l) ||
               (nd.typ[j] >= V2M_JUNCTION && nd.part[2 * j + 1] == l) || (nd.typ[j] == V2M_CORNER && nd.part3[j] == l);
    };

    for (int j = 0; j < n; ++j)   // junction / corner nodes: always kept
        if (nd.typ[j] >= V2M_JUNCTION) {
            insert(j);
        }

    std::vector<int> cand;
    cand.reserve(n);

    for (int i = 0; i < n; ++i)
        if (nd.typ[i] == V2M_INTERFACE || nd.typ[i] == V2M_INTERIOR) {
            cand.push_back(i);
        }

    std::sort(cand.begin(), cand.end(), [&](int a, int b) {
        const int ta = nd.typ[a] == V2M_INTERIOR, tb = nd.typ[b] == V2M_INTERIOR;   // interfaces first
        return ta != tb ? ta < tb : (hn[a] != hn[b] ? hn[a] < hn[b] : a < b);
    });
    size_t removed[2] = { 0, 0 };

    for (int i : cand) {
        const float r = B * hn[i], r2 = r * r;
        int L = 0;

        while (L + 1 < nlev && c0 * static_cast<float>(1 << L) < r) {
            ++L;
        }

        const float x = nd.P[3 * i], y = nd.P[3 * i + 1], z = nd.P[3 * i + 2];
        const bool iface = nd.typ[i] == V2M_INTERFACE;
        const int a = nd.lab[i], b = iface ? nd.part[2 * i] : -1;
        bool crowded = false;

        for (int dz = -1; dz <= 1 && !crowded; ++dz)
            for (int dy = -1; dy <= 1 && !crowded; ++dy)
                for (int dx = -1; dx <= 1 && !crowded; ++dx) {
                    auto it = grid.find(cell(L, x, y, z, dx, dy, dz));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (int j : it->second) {
                        const float ex = nd.P[3 * j] - x, ey = nd.P[3 * j + 1] - y, ez = nd.P[3 * j + 2] - z;

                        if (ex * ex + ey * ey + ez * ez >= r2) {
                            continue;
                        }

                        // same interface (a pair, on an interface / junction node), or the
                        // same label for an interior node
                        if (iface ? (nd.typ[j] != V2M_INTERIOR && has(j, a) && has(j, b)) : has(j, a)) {
                            crowded = true;
                            break;
                        }
                    }
                }

        if (crowded) {
            ++removed[iface ? 0 : 1];
        } else {
            insert(i);
        }
    }

    const size_t nrem = removed[0] + removed[1];

    if (nrem) {
        Nodes o;
        o.P.reserve(nd.P.size());

        for (int i = 0; i < n; ++i)
            if (keep[i]) {
                o.P.insert(o.P.end(), { nd.P[3 * i], nd.P[3 * i + 1], nd.P[3 * i + 2] });
                o.lab.push_back(nd.lab[i]);
                o.typ.push_back(nd.typ[i]);
                o.part.push_back(nd.part[2 * i]);
                o.part.push_back(nd.part[2 * i + 1]);
                o.part3.push_back(nd.part3[i]);
            }

        nd = std::move(o);
    }

    if (prm.verbose) {
        V2M_FPRINTF(stderr, "[thin]  %zu of %d nodes removed (interface %zu, interior %zu; B = %.2f)\n", nrem, n,
                   removed[0], removed[1], B);
    }

    return nrem;
}

}  // namespace tn
