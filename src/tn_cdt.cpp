// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (the CDT driver adapted from gpu_brain2mesh src/cdt/b2m_cdt_diazzi.cpp, same
//  author, GPL-3.0-or-later; the CDT itself: third_party/cdt, LGPL-3.0-or-later)
//
// tn_cdt.cpp -- see tn_cdt.h.

#include "tn_cdt.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tn_remesh.h"

#ifdef TN_HAS_CDT
// inputPLC.h wants its dependencies included first and defines non-inline free
// functions: this is the only translation unit that includes it (as in cdt's own
// main.cpp: delaunay.h, then inputPLC.h, then PLC.h)
#include <algorithm>
#include <cfloat>
#include <cstring>
#include "delaunay.h"
#include "inputPLC.h"
#include "PLC.h"
#endif

namespace tn {

void cdt_mesh(const Mesh& m, Mesh& out, CdtStats& st, double fill) {
#ifndef TN_HAS_CDT
    (void)fill;
    (void)m;
    (void)out;
    (void)st;
    throw std::runtime_error("cdt: built without the CDT (TN_USE_CDT=OFF)");
#else
    const auto t0 = std::chrono::steady_clock::now();
    const size_t nf = m.tris.size() / 3;

    if (nf == 0) {
        throw std::runtime_error("cdt: the input has no triangles");
    }

    // 1. closed? (an edge on one triangle is a hole; on 3+ a junction, fine)
    {
        std::unordered_map<uint64_t, int> ec;

        for (size_t t = 0; t < nf; ++t)
            for (int k = 0; k < 3; ++k) {
                uint64_t a = static_cast<uint32_t>(m.tris[3 * t + k]), b = static_cast<uint32_t>(m.tris[3 * t + (k + 1) % 3]);

                if (a > b) {
                    std::swap(a, b);
                }

                ++ec[(a << 32) | b];
            }

        for (const auto& kv : ec) {
            st.open_edges += kv.second == 1;
            st.junction_edges += kv.second > 2;
        }

        if (st.open_edges > 0) {
            throw std::runtime_error("cdt: the surface is not closed (" + std::to_string(st.open_edges) +
                                     " open edges); --mode repair closes it");
        }
    }

    // 2. the CDT
    std::vector<double> pts(m.nodes);
    std::vector<uint32_t> tri(m.tris.begin(), m.tris.end());
    inputPLC plc;
    plc.initFromVectors(pts.data(), static_cast<uint32_t>(pts.size() / 3), tri.data(), static_cast<uint32_t>(nf), false);
    st.plc_vertices = plc.numVertices();
    st.plc_triangles = plc.numTriangles();
    // interior points (fill): a body-centred cubic lattice inside the regions,
    // clear of the surface; appended after the PLC's vertices, so the constraint
    // triangles' indices still hold
    const RegionLocator loc(m);
    std::vector<double> V(plc.coordinates);

    if (fill > 0) {
        double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };

        for (size_t v = 0; v < m.nodes.size() / 3; ++v)
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min(lo[a], m.nodes[3 * v + a]);
                hi[a] = std::max(hi[a], m.nodes[3 * v + a]);
            }

        // the triangles binned in cells of the clearance, for the distance test
        const double clear = 0.4 * fill, cell = std::max(clear, 1e-300);
        std::unordered_map<int64_t, std::vector<int32_t>> bins;
        auto ckey = [&](int64_t i, int64_t j, int64_t k) {
            return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
        };

        for (size_t t = 0; t < nf; ++t) {
            int64_t b0[3], b1[3];

            for (int a = 0; a < 3; ++a) {
                double mn = 1e300, mx = -1e300;

                for (int k = 0; k < 3; ++k) {
                    const double x = m.nodes[3 * static_cast<size_t>(m.tris[3 * t + k]) + a];
                    mn = std::min(mn, x);
                    mx = std::max(mx, x);
                }

                b0[a] = static_cast<int64_t>(std::floor(mn / cell));
                b1[a] = static_cast<int64_t>(std::floor(mx / cell));
            }

            for (int64_t k = b0[2]; k <= b1[2]; ++k)
                for (int64_t j = b0[1]; j <= b1[1]; ++j)
                    for (int64_t i = b0[0]; i <= b1[0]; ++i) {
                        bins[ckey(i, j, k)].push_back(static_cast<int32_t>(t));
                    }
        }

        auto near_surface = [&](const double* p) {
            const int64_t ci = static_cast<int64_t>(std::floor(p[0] / cell)), cj = static_cast<int64_t>(std::floor(p[1] / cell)),
                          ck = static_cast<int64_t>(std::floor(p[2] / cell));

            for (int64_t dk = -1; dk <= 1; ++dk)
                for (int64_t dj = -1; dj <= 1; ++dj)
                    for (int64_t di = -1; di <= 1; ++di) {
                        auto it = bins.find(ckey(ci + di, cj + dj, ck + dk));

                        if (it == bins.end()) {
                            continue;
                        }

                        for (int32_t t : it->second) {
                            // distance to the triangle's plane, clamped to its corners' sphere: cheap and conservative
                            const double* a = &m.nodes[3 * static_cast<size_t>(m.tris[3 * t])];
                            const double* b = &m.nodes[3 * static_cast<size_t>(m.tris[3 * t + 1])];
                            const double* c = &m.nodes[3 * static_cast<size_t>(m.tris[3 * t + 2])];
                            double best = 1e300;

                            // the corners, the edge midpoints and the centroid: a sampled distance
                            const double* cs[3] = { a, b, c };

                            for (int u = 0; u < 3; ++u)
                                for (int w = u; w < 3; ++w) {
                                    const double q[3] = { 0.5 * (cs[u][0] + cs[w][0]), 0.5 * (cs[u][1] + cs[w][1]), 0.5 * (cs[u][2] + cs[w][2]) };
                                    best = std::min(best, (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
                                }

                            // and the plane, where the foot falls inside the triangle
                            const double e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
                            const double n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
                            const double nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

                            if (nn > 0) {
                                const double d = ((p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1] + (p[2] - a[2]) * n[2]);
                                const double f[3] = { p[0] - d * n[0] / nn, p[1] - d * n[1] / nn, p[2] - d * n[2] / nn };
                                auto side = [&](const double* u, const double* v) {
                                    const double g[3] = { v[0] - u[0], v[1] - u[1], v[2] - u[2] }, h[3] = { f[0] - u[0], f[1] - u[1], f[2] - u[2] };
                                    return (g[1] * h[2] - g[2] * h[1]) * n[0] + (g[2] * h[0] - g[0] * h[2]) * n[1] + (g[0] * h[1] - g[1] * h[0]) * n[2];
                                };

                                if (side(a, b) >= 0 && side(b, c) >= 0 && side(c, a) >= 0) {
                                    best = std::min(best, d * d / nn);
                                }
                            }

                            if (best < clear * clear) {
                                return true;
                            }
                        }
                    }

            return false;
        };

        for (int sub = 0; sub < 2; ++sub) {   // the two interleaved cubic lattices
            const double off = sub * 0.5 * fill;

            for (double z = lo[2] + off + 0.5 * fill; z < hi[2]; z += fill)
                for (double y = lo[1] + off + 0.5 * fill; y < hi[1]; y += fill)
                    for (double x = lo[0] + off + 0.5 * fill; x < hi[0]; x += fill) {
                        // a hair of deterministic jitter: no exactly cospherical lattice
                        const uint64_t h = static_cast<uint64_t>(V.size()) * 0x9E3779B97F4A7C15ULL;
                        const double p[3] = { x + fill * 1e-5 * static_cast<double>((h >> 11) & 1023) / 1023.0,
                                              y + fill * 1e-5 * static_cast<double>((h >> 23) & 1023) / 1023.0,
                                              z + fill * 1e-5 * static_cast<double>((h >> 37) & 1023) / 1023.0
                                            };

                        if (loc.label_at(p) > 0 && !near_surface(p)) {
                            V.insert(V.end(), { p[0], p[1], p[2] });
                            ++st.interior;
                        }
                    }
        }
    }

    TetMesh tin;
    tin.init_vertices(V.data(), static_cast<uint32_t>(V.size() / 3));
    tin.tetrahedrize();
    PLCx splc(tin, plc.triangle_vertices.data(), plc.numTriangles());
    splc.segmentRecovery_HSi(true);
    splc.faceRecovery(true);
    st.steiner = tin.numVertices() > V.size() / 3 ? tin.numVertices() - V.size() / 3 : 0;

    // the constraint faces (as tet corners)
    std::vector<bool> cmask(tin.tet_node.size(), false);

    for (size_t fi = 0; fi < splc.faces.size(); ++fi) {
        splc.getTetsIntersectingFace(static_cast<uint32_t>(fi), nullptr, &cmask);
    }

    // vertex positions (a recovery Steiner point is implicit: its approximation)
    const uint32_t nv = tin.numVertices();
    std::vector<double> X(static_cast<size_t>(nv) * 3);

    for (uint32_t v = 0; v < nv; ++v) {
        tin.vertices[v]->getApproxXYZCoordinates(X[3 * v], X[3 * v + 1], X[3 * v + 2]);
    }

    // 3. the compartments (ghost tets: the exterior, not flooded into)
    const uint64_t nt = tin.numTets();
    std::vector<int> comp(static_cast<size_t>(nt), -1);
    std::vector<uint64_t> big;       // per compartment: its largest tet
    std::vector<double> bigvol;
    std::vector<uint64_t> stk;

    auto vol6 = [&](uint64_t t) {
        const uint32_t* v = &tin.tet_node[t << 2];
        const double* a = &X[3 * static_cast<size_t>(v[0])], *b = &X[3 * static_cast<size_t>(v[1])];
        const double* c = &X[3 * static_cast<size_t>(v[2])], *d = &X[3 * static_cast<size_t>(v[3])];
        const double ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
        const double bx = c[0] - a[0], by = c[1] - a[1], bz = c[2] - a[2];
        const double cx = d[0] - a[0], cy = d[1] - a[1], cz = d[2] - a[2];
        return std::fabs(ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx));
    };

    for (uint64_t s0 = 0; s0 < nt; ++s0) {
        if (tin.isGhost(s0) || comp[s0] >= 0) {
            continue;
        }

        const int cid = static_cast<int>(big.size());
        big.push_back(s0);
        bigvol.push_back(-1);
        comp[s0] = cid;
        stk.assign(1, s0);

        while (!stk.empty()) {
            const uint64_t t = stk.back();
            stk.pop_back();
            const double v = vol6(t);

            if (v > bigvol[static_cast<size_t>(cid)]) {
                bigvol[static_cast<size_t>(cid)] = v;
                big[static_cast<size_t>(cid)] = t;
            }

            for (int j = 0; j < 4; ++j) {
                const uint64_t c = (t << 2) | static_cast<uint64_t>(j), nc = tin.tet_neigh[c], n2 = nc >> 2;

                if (tin.isGhost(n2) || comp[n2] >= 0 || cmask[c] || cmask[nc]) {
                    continue;   // the outside, done, or a constraint face
                }

                comp[n2] = cid;
                stk.push_back(n2);
            }
        }
    }

    st.compartments = big.size();

    // 4. a label per compartment: its largest tet's centroid among the regions
    std::vector<int> clab(big.size(), 0);

    for (size_t k = 0; k < big.size(); ++k) {
        const uint32_t* v = &tin.tet_node[big[k] << 2];
        double c[3] = { 0, 0, 0 };

        for (int i = 0; i < 4; ++i)
            for (int a = 0; a < 3; ++a) {
                c[a] += 0.25 * X[3 * static_cast<size_t>(v[i]) + a];
            }

        clab[k] = loc.label_at(c);
        st.kept_compartments += clab[k] > 0;
    }

    // the output: the kept tets, near-coincident vertices welded (a Steiner point
    // of nearly parallel constraints can land a hair from another vertex)
    double ext = 0;

    for (int a = 0; a < 3; ++a) {
        double lo = 1e300, hi = -1e300;

        for (uint32_t v = 0; v < nv; ++v) {
            lo = std::min(lo, X[3 * v + a]);
            hi = std::max(hi, X[3 * v + a]);
        }

        ext = std::max(ext, hi - lo);
    }

    const double tol = 1e-9 * std::max(ext, 1e-300);
    std::vector<int32_t> weld(nv), map(nv, -1);
    {
        std::unordered_map<int64_t, std::vector<uint32_t>> grid;
        auto key = [&](double x, double y, double z, int dx, int dy, int dz) {
            const int64_t i = static_cast<int64_t>(std::floor(x / tol)) + dx, j = static_cast<int64_t>(std::floor(y / tol)) + dy,
                          k = static_cast<int64_t>(std::floor(z / tol)) + dz;
            return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
        };

        for (uint32_t v = 0; v < nv; ++v) {
            const double x = X[3 * v], y = X[3 * v + 1], z = X[3 * v + 2];
            int32_t rep = -1;

            for (int dz = -1; dz <= 1 && rep < 0; ++dz)
                for (int dy = -1; dy <= 1 && rep < 0; ++dy)
                    for (int dx = -1; dx <= 1 && rep < 0; ++dx) {
                        auto it = grid.find(key(x, y, z, dx, dy, dz));

                        if (it == grid.end()) {
                            continue;
                        }

                        for (uint32_t u : it->second) {
                            const double ex = X[3 * u] - x, ey = X[3 * u + 1] - y, ez = X[3 * u + 2] - z;

                            if (ex * ex + ey * ey + ez * ez <= tol * tol) {
                                rep = static_cast<int32_t>(u);
                                break;
                            }
                        }
                    }

            if (rep < 0) {
                weld[v] = static_cast<int32_t>(v);
                grid[key(x, y, z, 0, 0, 0)].push_back(v);
            } else {
                weld[v] = rep;
                ++st.welded;
            }
        }
    }

    out = Mesh();

    for (uint64_t t = 0; t < nt; ++t) {
        if (tin.isGhost(t) || comp[t] < 0 || clab[static_cast<size_t>(comp[t])] <= 0) {
            continue;
        }

        int32_t w[4];

        for (int i = 0; i < 4; ++i) {
            w[i] = weld[tin.tet_node[(t << 2) + i]];
        }

        if (w[0] == w[1] || w[0] == w[2] || w[0] == w[3] || w[1] == w[2] || w[1] == w[3] || w[2] == w[3]) {
            ++st.degenerate;
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            if (map[w[i]] < 0) {
                map[w[i]] = static_cast<int32_t>(out.nodes.size() / 3);
                out.nodes.insert(out.nodes.end(), { X[3 * static_cast<size_t>(w[i])], X[3 * static_cast<size_t>(w[i]) + 1],
                                                    X[3 * static_cast<size_t>(w[i]) + 2] });
            }

            out.tets.push_back(map[w[i]]);
        }

        out.tet_labels.push_back(clab[static_cast<size_t>(comp[t])]);
    }

    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#endif
}

}  // namespace tn
