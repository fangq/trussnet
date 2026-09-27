// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_remesh.cpp -- see tn_remesh.h.

#include "tn_remesh.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "tn_omp.h"

namespace tn {

namespace {

typedef std::array<double, 3> V3;

inline V3 sub(const V3& a, const V3& b) {
    return { { a[0] - b[0], a[1] - b[1], a[2] - b[2] } };
}
inline double dot(const V3& a, const V3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline V3 cross(const V3& a, const V3& b) {
    return { { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] } };
}

// squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)
double dist2_tri(const V3& p, const V3& a, const V3& b, const V3& c) {
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    V3 q;

    if (d1 <= 0 && d2 <= 0) {
        q = a;
    } else {
        const V3 bp = sub(p, b);
        const double d3 = dot(ab, bp), d4 = dot(ac, bp);

        if (d3 >= 0 && d4 <= d3) {
            q = b;
        } else {
            const double vc = d1 * d4 - d3 * d2;

            if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                const double v = d1 / (d1 - d3);
                q = { { a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2] } };
            } else {
                const V3 cp = sub(p, c);
                const double d5 = dot(ab, cp), d6 = dot(ac, cp);

                if (d6 >= 0 && d5 <= d6) {
                    q = c;
                } else {
                    const double vb = d5 * d2 - d1 * d6;

                    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                        const double w = d2 / (d2 - d6);
                        q = { { a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2] } };
                    } else {
                        const double va = d3 * d6 - d5 * d4;

                        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            q = { { b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2]) } };
                        } else {
                            const double den = 1.0 / (va + vb + vc), v = vb * den, w = vc * den;
                            q = { { a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w } };
                        }
                    }
                }
            }
        }
    }

    const V3 d = sub(p, q);
    return dot(d, d);
}

// A set of oriented triangles (one region's or one shell's boundary) with a
// y-z bin grid for rays along +x: the winding number of a point is the sum of
// sign(n_x) over the faces the ray from it crosses (+1 inside an outward
// surface; 2 where two parts overlap).
struct Surf {
    std::vector<std::array<V3, 3>> tri;
    // bins
    double y0 = 0, z0 = 0, cell = 1;
    int by = 1, bz = 1;
    std::vector<std::vector<int32_t>> bin;

    void build() {
        double ylo = 1e300, yhi = -1e300, zlo = 1e300, zhi = -1e300, esum = 0;

        for (const auto& t : tri)
            for (int k = 0; k < 3; ++k) {
                ylo = std::min(ylo, t[k][1]);
                yhi = std::max(yhi, t[k][1]);
                zlo = std::min(zlo, t[k][2]);
                zhi = std::max(zhi, t[k][2]);
                const V3 e = sub(t[(k + 1) % 3], t[k]);
                esum += std::sqrt(dot(e, e));
            }

        cell = std::max(tri.empty() ? 1.0 : 1.5 * esum / (3.0 * tri.size()), 1e-9);
        y0 = ylo;
        z0 = zlo;
        by = std::max(1, std::min(2048, static_cast<int>((yhi - ylo) / cell) + 1));
        bz = std::max(1, std::min(2048, static_cast<int>((zhi - zlo) / cell) + 1));
        bin.assign(static_cast<size_t>(by) * bz, {});

        for (size_t i = 0; i < tri.size(); ++i) {
            const auto& t = tri[i];
            const double mny = std::min(t[0][1], std::min(t[1][1], t[2][1])), mxy = std::max(t[0][1], std::max(t[1][1], t[2][1]));
            const double mnz = std::min(t[0][2], std::min(t[1][2], t[2][2])), mxz = std::max(t[0][2], std::max(t[1][2], t[2][2]));
            const int j0 = clampi(static_cast<int>((mny - y0) / cell), by), j1 = clampi(static_cast<int>((mxy - y0) / cell), by);
            const int k0 = clampi(static_cast<int>((mnz - z0) / cell), bz), k1 = clampi(static_cast<int>((mxz - z0) / cell), bz);

            for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j) {
                    bin[static_cast<size_t>(k) * by + j].push_back(static_cast<int32_t>(i));
                }
        }
    }
    static int clampi(int v, int n) {
        return v < 0 ? 0 : v >= n ? n - 1 : v;
    }
    // the crossings of the ray through (y, z) along x: (x, sign), unsorted
    void crossings(double y, double z, std::vector<std::pair<double, int>>& out) const {
        out.clear();
        const int j = static_cast<int>((y - y0) / cell), k = static_cast<int>((z - z0) / cell);

        if (j < 0 || j >= by || k < 0 || k >= bz) {
            return;
        }

        for (int32_t i : bin[static_cast<size_t>(k) * by + j]) {
            const auto& t = tri[static_cast<size_t>(i)];
            // (y, z) inside the projection: the three edge functions of one sign
            double e[3];

            for (int a = 0; a < 3; ++a) {
                const V3& p = t[a], &q = t[(a + 1) % 3];
                e[a] = (q[1] - p[1]) * (z - p[2]) - (q[2] - p[2]) * (y - p[1]);
            }

            const bool pos = e[0] > 0 && e[1] > 0 && e[2] > 0, neg = e[0] < 0 && e[1] < 0 && e[2] < 0;

            if (!pos && !neg) {
                continue;
            }

            const double s = e[0] + e[1] + e[2];   // = twice the signed projected area
            // barycentric weights: e[a] weights the vertex opposite edge a
            const double x = (e[1] * t[0][0] + e[2] * t[1][0] + e[0] * t[2][0]) / s;
            out.emplace_back(x, pos ? 1 : -1);   // pos: the normal has n_x > 0
        }
    }
    int winding(const V3& p) const {
        // the ray nudged off any edge / vertex it would graze (a query at a mesh
        // vertex, on a diagonal: the crossing would be missed)
        std::vector<std::pair<double, int>> c;
        crossings(p[1] + 1.37e-7 * cell, p[2] + 2.71e-7 * cell, c);
        int w = 0;

        for (const auto& h : c)
            if (h.first > p[0]) {
                w += h.second;
            }

        return w;
    }
};

// consistent, outward orientation per connected component (edge-adjacent
// through manifold edges); returns the faces flipped. `comp` gets the component
// of each face.
size_t orient_shells(std::vector<std::array<int32_t, 3>>& f, const std::vector<double>& X, std::vector<int>& comp) {
    const size_t nf = f.size();
    std::unordered_map<uint64_t, std::vector<int32_t>> edges;

    for (size_t i = 0; i < nf; ++i)
        for (int k = 0; k < 3; ++k) {
            uint64_t a = static_cast<uint32_t>(f[i][k]), b = static_cast<uint32_t>(f[i][(k + 1) % 3]);

            if (a > b) {
                std::swap(a, b);
            }

            edges[(a << 32) | b].push_back(static_cast<int32_t>(i));
        }

    comp.assign(nf, -1);
    size_t flipped = 0;
    int nc = 0;
    std::vector<int32_t> stack;

    auto has_edge = [](const std::array<int32_t, 3>& t, int32_t a, int32_t b) {   // directed a -> b
        for (int k = 0; k < 3; ++k)
            if (t[k] == a && t[(k + 1) % 3] == b) {
                return true;
            }

        return false;
    };

    for (size_t s = 0; s < nf; ++s) {
        if (comp[s] >= 0) {
            continue;
        }

        const int c = nc++;
        comp[s] = c;
        stack.assign(1, static_cast<int32_t>(s));
        std::vector<int32_t> members;

        while (!stack.empty()) {
            const int32_t i = stack.back();
            stack.pop_back();
            members.push_back(i);

            for (int k = 0; k < 3; ++k) {
                const int32_t a = f[static_cast<size_t>(i)][k], b = f[static_cast<size_t>(i)][(k + 1) % 3];
                uint64_t ea = static_cast<uint32_t>(a), eb = static_cast<uint32_t>(b);

                if (ea > eb) {
                    std::swap(ea, eb);
                }

                const std::vector<int32_t>& nb = edges[(ea << 32) | eb];

                for (int32_t j : nb) {
                    if (j == i || comp[static_cast<size_t>(j)] >= 0) {
                        continue;
                    }

                    comp[static_cast<size_t>(j)] = c;

                    if (nb.size() == 2 && has_edge(f[static_cast<size_t>(j)], a, b)) {   // same direction: flip j
                        std::swap(f[static_cast<size_t>(j)][1], f[static_cast<size_t>(j)][2]);
                        ++flipped;
                    }

                    stack.push_back(j);
                }
            }
        }

        // outward: positive signed volume
        double vol = 0;

        for (int32_t i : members) {
            const double* a = &X[3 * static_cast<size_t>(f[static_cast<size_t>(i)][0])];
            const double* b = &X[3 * static_cast<size_t>(f[static_cast<size_t>(i)][1])];
            const double* cc = &X[3 * static_cast<size_t>(f[static_cast<size_t>(i)][2])];
            vol += a[0] * (b[1] * cc[2] - b[2] * cc[1]) - a[1] * (b[0] * cc[2] - b[2] * cc[0]) + a[2] * (b[0] * cc[1] - b[1] * cc[0]);
        }

        if (vol < 0) {
            for (int32_t i : members) {
                std::swap(f[static_cast<size_t>(i)][1], f[static_cast<size_t>(i)][2]);
            }

            flipped += members.size();
        }
    }

    return flipped;
}

// the regions of a surface (tn_remesh.h): each label's outward boundary, or
// the shells with their labels and nesting
struct Regions {
    bool pairs = false;
    std::map<int, Surf> region;
    std::vector<Surf> shell;
    std::vector<int> shell_label, shell_parent, shell_depth;
};

void build_regions(const Mesh& m, Regions& R, RasterStats& st) {
    const size_t nf = m.tris.size() / 3;

    if (nf == 0) {
        throw std::runtime_error("the input has no triangles");
    }

    st.faces = nf;
    const std::vector<double>& X = m.nodes;
    std::vector<std::array<int32_t, 3>> f(nf);

    for (size_t i = 0; i < nf; ++i) {
        f[i] = { { m.tris[3 * i], m.tris[3 * i + 1], m.tris[3 * i + 2] } };
    }

    auto P = [&](int32_t v) {
        return V3{ { X[3 * static_cast<size_t>(v)], X[3 * static_cast<size_t>(v) + 1], X[3 * static_cast<size_t>(v) + 2] } };
    };

    // the regions: (label, its outward-oriented boundary)
    std::map<int, Surf>& region = R.region;
    // inner / outer label pairs: some face lies between two tissues (or its
    // outer label is not the exterior)
    bool& pairs = R.pairs;
    pairs = false;

    if (m.tri_labels.size() == 2 * nf)
        for (size_t i = 0; i < nf && !pairs; ++i) {
            pairs = m.tri_labels[2 * i + 1] != 0;
        }

    // nesting for shells: each shell's own surface and label
    std::vector<Surf>& shell = R.shell;
    std::vector<int>& shell_label = R.shell_label;
    std::vector<int>& shell_parent = R.shell_parent;

    if (pairs) {
        // region l: faces with inner = l as given (normal inner -> outer, i.e.
        // outward from l), faces with outer = l flipped
        for (size_t i = 0; i < nf; ++i) {
            const int a = m.tri_labels[2 * i], b = m.tri_labels[2 * i + 1];
            const std::array<V3, 3> t = { { P(f[i][0]), P(f[i][1]), P(f[i][2]) } };

            if (a > 0) {
                region[a].tri.push_back(t);
            }

            if (b > 0 && b != a) {
                region[b].tri.push_back({ { t[0], t[2], t[1] } });
            }
        }
    } else {
        std::vector<int> comp;
        st.flipped = orient_shells(f, X, comp);
        const int nc = comp.empty() ? 0 : *std::max_element(comp.begin(), comp.end()) + 1;
        shell.resize(static_cast<size_t>(nc));
        std::vector<std::map<int, size_t>> votes(static_cast<size_t>(nc));

        for (size_t i = 0; i < nf; ++i) {
            shell[static_cast<size_t>(comp[i])].tri.push_back({ { P(f[i][0]), P(f[i][1]), P(f[i][2]) } });

            if (m.tri_labels.size() == 2 * nf) {
                ++votes[static_cast<size_t>(comp[i])][m.tri_labels[2 * i]];
            }
        }

        for (Surf& s : shell) {
            s.build();
        }

        // nesting: shell a contains shell b if (nearly) all of b's corners are
        // inside a (winding >= 1) -- a shell that only overlaps another is not
        // nested in it (unlabelled, it gets the same label: their union); depth =
        // the number of shells containing it; parent = the deepest of those
        std::vector<int> depth(static_cast<size_t>(nc), 0);
        shell_parent.assign(static_cast<size_t>(nc), -1);
        std::vector<std::vector<int>> within(static_cast<size_t>(nc));

        for (int b = 0; b < nc; ++b) {
            const Surf& B = shell[static_cast<size_t>(b)];
            const size_t ns = std::min<size_t>(B.tri.size(), 64), step = std::max<size_t>(1, B.tri.size() / ns);

            for (int a = 0; a < nc; ++a) {
                if (a == b) {
                    continue;
                }

                size_t in = 0, tot = 0;

                for (size_t i = 0; i < B.tri.size() && tot < ns; i += step, ++tot) {
                    in += shell[static_cast<size_t>(a)].winding(B.tri[i][0]) >= 1;
                }

                if (tot > 0 && in * 100 >= tot * 98) {
                    within[static_cast<size_t>(b)].push_back(a);
                    ++depth[static_cast<size_t>(b)];
                }
            }
        }

        for (int b = 0; b < nc; ++b)
            for (int a : within[static_cast<size_t>(b)])
                if (shell_parent[static_cast<size_t>(b)] < 0 || depth[static_cast<size_t>(a)] > depth[static_cast<size_t>(shell_parent[static_cast<size_t>(b)])]) {
                    shell_parent[static_cast<size_t>(b)] = a;
                }

        shell_label.resize(static_cast<size_t>(nc));

        for (int c = 0; c < nc; ++c) {
            int lab = depth[static_cast<size_t>(c)] + 1;
            size_t best = 0;

            for (const auto& kv : votes[static_cast<size_t>(c)])
                if (kv.first > 0 && kv.second > best) {
                    best = kv.second;
                    lab = kv.first;
                }

            shell_label[static_cast<size_t>(c)] = lab;
        }

        st.shells = nc;
        R.shell_depth = depth;
    }

    for (auto& kv : region) {
        kv.second.build();
    }
}

}  // namespace

Tpm rasterize_surfaces(const Mesh& m, double voxel, RasterStats& st) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t nf = m.tris.size() / 3;

    if (nf == 0) {
        throw std::runtime_error("remesh: the input has no triangles");
    }

    if (!(voxel > 0)) {
        throw std::runtime_error("remesh: the raster voxel must be > 0");
    }

    Regions R;
    build_regions(m, R, st);
    const std::vector<double>& X = m.nodes;
    const bool pairs = R.pairs;
    std::map<int, Surf>& region = R.region;
    std::vector<Surf>& shell = R.shell;
    std::vector<int>& shell_label = R.shell_label;
    std::vector<int>& shell_parent = R.shell_parent;

    // the raster: the bounding box + a margin
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };

    for (size_t v = 0; v < X.size() / 3; ++v)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], X[3 * v + k]);
            hi[k] = std::max(hi[k], X[3 * v + k]);
        }

    const int margin = 4;
    const double band = 3.0 * voxel;   // exact distances within this of a boundary face
    Tpm t;
    t.nx = static_cast<int>(std::ceil((hi[0] - lo[0]) / voxel)) + 1 + 2 * margin;
    t.ny = static_cast<int>(std::ceil((hi[1] - lo[1]) / voxel)) + 1 + 2 * margin;
    t.nz = static_cast<int>(std::ceil((hi[2] - lo[2]) / voxel)) + 1 + 2 * margin;

    if (static_cast<double>(t.nx) * t.ny * t.nz > 1.2e9) {
        throw std::runtime_error("remesh: the raster would be too large; use a larger --raster-voxel");
    }

    const double o[3] = { lo[0] - margin * voxel, lo[1] - margin * voxel, lo[2] - margin * voxel };
    t.voxelsize = { { voxel, voxel, voxel } };
    t.affine = { { voxel, 0, 0, o[0], 0, voxel, 0, o[1], 0, 0, voxel, o[2], 0, 0, 0, 1 } };
    st.nx = t.nx;
    st.ny = t.ny;
    st.nz = t.nz;
    st.voxel = voxel;
    const size_t nv = t.nv();
    const int nx = t.nx, ny = t.ny, nz = t.nz;

    // the signed distance of one surface (> 0 inside), from its winding number
    // and the distance to its boundary faces
    auto signed_distance = [&](const Surf& S, std::vector<float>& sd) {
        // inside, row by row (voxel centres, the ray nudged off the grid lines)
        std::vector<uint8_t> in(nv, 0);
        #pragma omp parallel for schedule(dynamic, 8)

        for (int64_t r = 0; r < static_cast<int64_t>(ny) * nz; ++r) {
            const int j = static_cast<int>(r % ny), k = static_cast<int>(r / ny);
            const double y = o[1] + (j + 1.3e-6) * voxel, z = o[2] + (k + 2.7e-6) * voxel;
            std::vector<std::pair<double, int>> c;
            S.crossings(y, z, c);

            if (c.empty()) {
                continue;
            }

            std::sort(c.begin(), c.end());
            // w(x) = sum of the signs of the crossings beyond x: sweep from the right
            int w = 0;
            size_t h = c.size();

            for (int i = nx - 1; i >= 0; --i) {
                const double x = o[0] + i * voxel;

                while (h > 0 && c[h - 1].first > x) {
                    w += c[--h].second;
                }

                in[static_cast<size_t>(k) * nx * ny + static_cast<size_t>(j) * nx + i] = w >= 1 ? 1 : 0;
            }
        }

        // the boundary: where the inside differs on the two sides of a face (tested
        // at sample points off by a hair). A face exposed at all its samples is
        // kept whole, one buried at all of them dropped; a mixed one -- partly
        // buried where surfaces cross or overlap -- is split 1 -> 4 and its parts
        // tested again, down to about a voxel, so only the exposed part is kept
        const double eps = 1e-4 * voxel;
        auto exposed = [&](const std::array<V3, 3>& q, const V3& at) {
            V3 n = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            const double ln = std::sqrt(dot(n, n));

            if (!(ln > 0)) {
                return false;
            }

            const V3 sp = { { at[0] + eps * n[0] / ln, at[1] + eps * n[1] / ln, at[2] + eps * n[2] / ln } };
            const V3 sm = { { at[0] - eps * n[0] / ln, at[1] - eps * n[1] / ln, at[2] - eps * n[2] / ln } };
            return (S.winding(sp) >= 1) != (S.winding(sm) >= 1);
        };
        std::vector<std::vector<std::array<V3, 3>>> kept_of(S.tri.size());
        #pragma omp parallel for schedule(dynamic, 256)

        for (int64_t i = 0; i < static_cast<int64_t>(S.tri.size()); ++i) {
            std::vector<std::array<V3, 3>> todo(1, S.tri[static_cast<size_t>(i)]);
            std::vector<std::array<V3, 3>>& out = kept_of[static_cast<size_t>(i)];

            while (!todo.empty()) {
                const std::array<V3, 3> q = todo.back();
                todo.pop_back();
                const V3 g = { { (q[0][0] + q[1][0] + q[2][0]) / 3, (q[0][1] + q[1][1] + q[2][1]) / 3, (q[0][2] + q[1][2] + q[2][2]) / 3 } };
                double lmax = 0;

                for (int a = 0; a < 3; ++a) {
                    const V3 e = sub(q[(a + 1) % 3], q[a]);
                    lmax = std::max(lmax, dot(e, e));
                }

                if (lmax <= voxel * voxel) {   // small: its centroid decides
                    if (exposed(q, g)) {
                        out.push_back(q);
                    }

                    continue;
                }

                int ex = exposed(q, g) ? 1 : 0;

                for (int a = 0; a < 3; ++a) {
                    const V3 s3 = { { 0.5 * (g[0] + q[a][0]), 0.5 * (g[1] + q[a][1]), 0.5 * (g[2] + q[a][2]) } };
                    ex += exposed(q, s3) ? 1 : 0;
                }

                if (ex == 4) {
                    out.push_back(q);
                } else if (ex > 0) {   // mixed: split
                    V3 m01, m12, m20;

                    for (int c = 0; c < 3; ++c) {
                        m01[c] = 0.5 * (q[0][c] + q[1][c]);
                        m12[c] = 0.5 * (q[1][c] + q[2][c]);
                        m20[c] = 0.5 * (q[2][c] + q[0][c]);
                    }

                    todo.push_back({ { q[0], m01, m20 } });
                    todo.push_back({ { m01, q[1], m12 } });
                    todo.push_back({ { m20, m12, q[2] } });
                    todo.push_back({ { m01, m12, m20 } });
                }
            }
        }

        std::vector<std::array<V3, 3>> piece;

        for (auto& v : kept_of) {
            piece.insert(piece.end(), v.begin(), v.end());
        }

        std::vector<char> keep(piece.size(), 1);

        // unsigned distance to the boundary faces in a band, by z plane
        std::vector<std::vector<int32_t>> plane(static_cast<size_t>(nz));

        for (size_t i = 0; i < piece.size(); ++i) {
            if (!keep[i]) {
                continue;
            }

            const auto& q = piece[i];
            const double mn = std::min(q[0][2], std::min(q[1][2], q[2][2])) - band, mx = std::max(q[0][2], std::max(q[1][2], q[2][2])) + band;
            const int k0 = std::max(0, static_cast<int>(std::floor((mn - o[2]) / voxel))), k1 = std::min(nz - 1, static_cast<int>(std::ceil((mx - o[2]) / voxel)));

            for (int k = k0; k <= k1; ++k) {
                plane[static_cast<size_t>(k)].push_back(static_cast<int32_t>(i));
            }
        }

        st.boundary_faces += piece.size();
        std::vector<float> d(nv, static_cast<float>(band));
        #pragma omp parallel for schedule(dynamic, 1)

        for (int k = 0; k < nz; ++k) {
            const double z = o[2] + k * voxel;

            for (int32_t i : plane[static_cast<size_t>(k)]) {
                const auto& q = piece[static_cast<size_t>(i)];
                const double mnx = std::min(q[0][0], std::min(q[1][0], q[2][0])) - band, mxx = std::max(q[0][0], std::max(q[1][0], q[2][0])) + band;
                const double mny = std::min(q[0][1], std::min(q[1][1], q[2][1])) - band, mxy = std::max(q[0][1], std::max(q[1][1], q[2][1])) + band;
                const int i0 = std::max(0, static_cast<int>(std::floor((mnx - o[0]) / voxel))), i1 = std::min(nx - 1, static_cast<int>(std::ceil((mxx - o[0]) / voxel)));
                const int j0 = std::max(0, static_cast<int>(std::floor((mny - o[1]) / voxel))), j1 = std::min(ny - 1, static_cast<int>(std::ceil((mxy - o[1]) / voxel)));

                for (int j = j0; j <= j1; ++j)
                    for (int ii = i0; ii <= i1; ++ii) {
                        const V3 p = { { o[0] + ii * voxel, o[1] + j * voxel, z } };
                        const float dd = static_cast<float>(std::sqrt(dist2_tri(p, q[0], q[1], q[2])));
                        float& cur = d[static_cast<size_t>(k) * nx * ny + static_cast<size_t>(j) * nx + ii];

                        if (dd < cur) {
                            cur = dd;
                        }
                    }
            }
        }

        sd.resize(nv);

        for (size_t v = 0; v < nv; ++v) {
            sd[v] = in[v] ? d[v] : -d[v];
        }
    };

    // the per-label signed distances
    std::map<int, std::vector<float>> lab_sd;

    if (pairs) {
        for (auto& kv : region) {
            signed_distance(kv.second, lab_sd[kv.first]);
        }
    } else {
        const int nc = static_cast<int>(shell.size());
        std::vector<std::vector<float>> sh(static_cast<size_t>(nc));

        for (int c = 0; c < nc; ++c) {
            signed_distance(shell[static_cast<size_t>(c)], sh[static_cast<size_t>(c)]);
        }

        // region of shell c = inside c, outside its children: min(s_c, -s_child);
        // shells of one label: the union (max)
        for (int c = 0; c < nc; ++c) {
            std::vector<float> r = sh[static_cast<size_t>(c)];

            for (int ch = 0; ch < nc; ++ch)
                if (shell_parent[static_cast<size_t>(ch)] == c)
                    for (size_t v = 0; v < nv; ++v) {
                        r[v] = std::min(r[v], -sh[static_cast<size_t>(ch)][v]);
                    }

            auto it = lab_sd.find(shell_label[static_cast<size_t>(c)]);

            if (it == lab_sd.end()) {
                lab_sd[shell_label[static_cast<size_t>(c)]] = r;
            } else
                for (size_t v = 0; v < nv; ++v) {
                    it->second[v] = std::max(it->second[v], r[v]);
                }
        }
    }

    // the map: channel 0 the exterior, channel l region l; p = clamp(0.5 + s / w)
    const int maxlab = lab_sd.empty() ? 0 : lab_sd.rbegin()->first;
    t.C = maxlab + 1;
    t.p.assign(static_cast<size_t>(t.C) * nv, 0.0f);
    t.names.assign(static_cast<size_t>(t.C), "");
    t.names[0] = "background";
    const double w = 1.5 * voxel;
    std::vector<float> smax(nv, -static_cast<float>(band));

    for (const auto& kv : lab_sd) {
        float* pl = &t.p[static_cast<size_t>(kv.first) * nv];

        for (size_t v = 0; v < nv; ++v) {
            pl[v] = static_cast<float>(std::min(1.0, std::max(0.0, 0.5 + kv.second[v] / w)));
            smax[v] = std::max(smax[v], kv.second[v]);
        }
    }

    for (size_t v = 0; v < nv; ++v) {
        t.p[v] = static_cast<float>(std::min(1.0, std::max(0.0, 0.5 - smax[v] / w)));
    }

    st.regions = static_cast<int>(lab_sd.size());
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return t;
}

struct RegionLocator::Impl {
    Regions R;
};

RegionLocator::RegionLocator(const Mesh& m) : impl_(new Impl) {
    RasterStats st;
    build_regions(m, impl_->R, st);
}

RegionLocator::~RegionLocator() = default;

int RegionLocator::label_at(const double* p) const {
    const Regions& R = impl_->R;
    const V3 q = { { p[0], p[1], p[2] } };

    if (R.pairs) {   // the region whose boundary winds around p (the largest label if several overlap)
        int lab = 0;

        for (const auto& kv : R.region)
            if (kv.second.winding(q) >= 1) {
                lab = kv.first;
            }

        return lab;
    }

    // shells: the innermost (deepest) one containing p
    int best = -1;

    for (size_t c = 0; c < R.shell.size(); ++c)
        if (R.shell[c].winding(q) >= 1 && (best < 0 || R.shell_depth[c] > R.shell_depth[static_cast<size_t>(best)])) {
            best = static_cast<int>(c);
        }

    return best < 0 ? 0 : R.shell_label[static_cast<size_t>(best)];
}

}  // namespace tn
