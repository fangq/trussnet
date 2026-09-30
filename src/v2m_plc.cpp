// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_plc.cpp -- see v2m_plc.h.

#include "v2m_plc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "delaunay.h"   // the exact orient2d of the vendored predicates
#include "v2m_2d.h"     // delaunay2d

namespace tn {

namespace {

// the file's lines, comments (# to the end) and blank lines dropped, as tokens
struct Lines {
    std::vector<std::vector<std::string>> L;
    std::vector<std::string> C;   // each line's comment
    size_t at = 0;
    std::string path;

    explicit Lines(const std::string& p) : path(p) {
        std::ifstream f(p);

        if (!f) {
            throw std::runtime_error(p + ": cannot open");
        }

        std::string s;

        while (std::getline(f, s)) {
            const size_t h = s.find('#');
            std::string c;

            if (h != std::string::npos) {
                c = s.substr(h + 1);
                s.erase(h);
            }

            std::istringstream is(s);
            std::vector<std::string> t;
            std::string w;

            while (is >> w) {
                t.push_back(w);
            }

            if (!t.empty()) {
                L.push_back(t);
                C.push_back(c);
            }
        }
    }
    bool more() const {
        return at < L.size();
    }
    const std::vector<std::string>& next(const char* what) {
        if (at >= L.size()) {
            throw std::runtime_error(path + ": ends before " + what);
        }

        return L[at++];
    }
    // a polygon: its count, then that many corners -- on as many lines as they take
    // (as TetGen reads them: a long list wraps)
    std::vector<std::string> counted(const char* what) {
        std::vector<std::string> t = next(what);
        const long c = std::strtol(t[0].c_str(), nullptr, 10);

        while (static_cast<long>(t.size()) < 1 + c) {
            const std::vector<std::string>& u = next(what);
            t.insert(t.end(), u.begin(), u.end());
        }

        return t;
    }
};

long to_long(const std::string& s, const std::string& path) {
    char* e = nullptr;
    const long v = std::strtol(s.c_str(), &e, 10);

    if (e == s.c_str() || *e) {
        throw std::runtime_error(path + ": not an integer: " + s);
    }

    return v;
}

double to_double(const std::string& s, const std::string& path) {
    char* e = nullptr;
    const double v = std::strtod(s.c_str(), &e);

    if (e == s.c_str() || *e) {
        throw std::runtime_error(path + ": not a number: " + s);
    }

    return v;
}

// a node list (part 1, or a .node file): the points, and their indices -> rows
void read_nodes(Lines& in, std::vector<double>& X, std::unordered_map<long, int32_t>& row, long& base) {
    const auto& h = in.next("the node list");
    const long n = to_long(h[0], in.path);

    if (h.size() > 1 && to_long(h[1], in.path) != 3) {
        throw std::runtime_error(in.path + ": only 3-D points (dimension 3)");
    }

    for (long i = 0; i < n; ++i) {
        const auto& t = in.next("the points");

        if (t.size() < 4) {
            throw std::runtime_error(in.path + ": a point line needs an index and x y z");
        }

        const long id = to_long(t[0], in.path);

        if (i == 0) {
            base = id;
        }

        row[id] = static_cast<int32_t>(X.size() / 3);

        for (int a = 0; a < 3; ++a) {
            X.push_back(to_double(t[static_cast<size_t>(1 + a)], in.path));
        }
    }
}

uint64_t ekey(int a, int b) {
    if (a > b) {
        std::swap(a, b);
    }

    return static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32 | static_cast<uint32_t>(b);
}

// A facet's constrained triangulation, in its plane (2-D, one axis dropped):
// Delaunay of its points (the enclosing triangle kept, so every edge lies
// inside), each polygon edge recovered by flips (Sloan), then the triangles
// reached from the enclosing triangle or a hole point without crossing an edge
// dropped. Returns the kept triangles (local point indices).
struct FacetCDT {
    std::vector<double> X;               // 2 per point (then the 3 enclosing corners)
    std::vector<int> T;                  // 3 per triangle, ccw; -1: removed
    std::unordered_map<uint64_t, std::array<int, 2>> E;   // edge -> its (up to 2) triangles
    std::unordered_map<uint64_t, char> cons;              // the constraint edges
    int n = 0;                            // the real points
    std::string where;

    int o2(int a, int b, int c) const {
        return orient2d(X[2 * a], X[2 * a + 1], X[2 * b], X[2 * b + 1], X[2 * c], X[2 * c + 1]);
    }
    void link(int t) {
        for (int k = 0; k < 3; ++k) {
            const uint64_t e = ekey(T[3 * t + k], T[3 * t + (k + 1) % 3]);
            const auto it = E.find(e);

            if (it == E.end()) {
                E[e] = { { t, -1 } };
            } else {
                it->second[1] = t;
            }
        }
    }
    void unlink(int t) {
        for (int k = 0; k < 3; ++k) {
            auto it = E.find(ekey(T[3 * t + k], T[3 * t + (k + 1) % 3]));

            if (it == E.end()) {
                continue;
            }

            auto& s = it->second;

            if (s[0] == t) {
                s[0] = s[1];
                s[1] = -1;
            } else if (s[1] == t) {
                s[1] = -1;
            }

            if (s[0] < 0) {
                E.erase(it);
            }
        }
    }
    bool has(int a, int b) const {
        return E.count(ekey(a, b)) > 0;
    }
    int apex(int t, int a, int b) const {
        for (int k = 0; k < 3; ++k)
            if (T[3 * t + k] != a && T[3 * t + k] != b) {
                return T[3 * t + k];
            }

        return -1;
    }
    void set(int t, int a, int b, int c) {   // ccw
        if (o2(a, b, c) < 0) {
            std::swap(b, c);
        }

        T[3 * t] = a;
        T[3 * t + 1] = b;
        T[3 * t + 2] = c;
    }
    // flip edge (u, v) if its quad is strictly convex; the new edge's ends
    bool flip(int u, int v, int& p, int& q) {
        const auto it = E.find(ekey(u, v));

        if (it == E.end() || it->second[1] < 0) {
            return false;
        }

        const int t1 = it->second[0], t2 = it->second[1];
        p = apex(t1, u, v);
        q = apex(t2, u, v);

        if (o2(p, q, u) * o2(p, q, v) >= 0) {
            return false;   // not strictly convex
        }

        unlink(t1);
        unlink(t2);
        set(t1, p, q, u);
        set(t2, p, q, v);
        link(t1);
        link(t2);
        return true;
    }
    bool crosses(int a, int b, int u, int v) const {   // the open segments cross
        return o2(a, b, u) * o2(a, b, v) < 0 && o2(u, v, a) * o2(u, v, b) < 0;
    }

    void insert(int a0, int b0) {
        std::deque<std::pair<int, int>> todo{ { a0, b0 } };

        while (!todo.empty()) {
            const int a = todo.front().first, b = todo.front().second;
            todo.pop_front();

            if (a == b) {
                continue;
            }

            if (has(a, b)) {
                cons[ekey(a, b)] = 1;
                continue;
            }

            // a point on the open segment: split there
            int on = -1;

            for (int c = 0; c < n && on < 0; ++c)
                if (c != a && c != b && o2(a, b, c) == 0) {
                    const double d = (X[2 * c] - X[2 * a]) * (X[2 * b] - X[2 * a]) +
                                     (X[2 * c + 1] - X[2 * a + 1]) * (X[2 * b + 1] - X[2 * a + 1]);
                    const double L = (X[2 * b] - X[2 * a]) * (X[2 * b] - X[2 * a]) +
                                     (X[2 * b + 1] - X[2 * a + 1]) * (X[2 * b + 1] - X[2 * a + 1]);

                    if (d > 0 && d < L) {
                        on = c;
                    }
                }

            if (on >= 0) {
                todo.emplace_front(on, b);
                todo.emplace_front(a, on);
                continue;
            }

            // the edges crossing it, flipped away (Sloan)
            std::deque<std::pair<int, int>> x;

            for (const auto& kv : E) {
                const int u = static_cast<int>(kv.first >> 32), v = static_cast<int>(kv.first & 0xffffffffu);

                if (crosses(a, b, u, v)) {
                    if (cons.count(kv.first)) {
                        throw std::runtime_error(where + ": two polygon edges cross");
                    }

                    x.emplace_back(u, v);
                }
            }

            const size_t limit = 64 * (x.size() + 16) * (x.size() + 16);   // (fixed at the start)

            for (size_t guard = 0; !x.empty(); ++guard) {
                if (guard > limit) {
                    throw std::runtime_error(where + ": a polygon edge could not be recovered");
                }

                const int u = x.front().first, v = x.front().second;
                x.pop_front();

                if (!has(u, v) || !crosses(a, b, u, v)) {
                    continue;
                }

                int p, q;

                if (!flip(u, v, p, q)) {
                    x.emplace_back(u, v);   // (not convex yet: later)
                } else if (crosses(a, b, p, q)) {
                    x.emplace_back(p, q);
                }
            }

            if (!has(a, b)) {
                throw std::runtime_error(where + ": a polygon edge could not be recovered");
            }

            cons[ekey(a, b)] = 1;
        }
    }

    // flood from triangle s across the non-constraint edges, marking out
    void flood(int s, std::vector<char>& out) const {
        std::vector<int> stk{ s };
        out[static_cast<size_t>(s)] = 1;

        while (!stk.empty()) {
            const int t = stk.back();
            stk.pop_back();

            for (int k = 0; k < 3; ++k) {
                const uint64_t e = ekey(T[3 * t + k], T[3 * t + (k + 1) % 3]);

                if (cons.count(e)) {
                    continue;
                }

                const auto& s2 = E.at(e);
                const int u = s2[0] == t ? s2[1] : s2[0];

                if (u >= 0 && !out[static_cast<size_t>(u)]) {
                    out[static_cast<size_t>(u)] = 1;
                    stk.push_back(u);
                }
            }
        }
    }
};

// the triangles of one facet: polygons (node rows), hole points (3-D)
// (h > 0: interior points on a grid of that spacing, clear of the boundary, so the
// triangles are not the polygon's long fan; new nodes appended to P)
void triangulate_facet(const std::vector<std::vector<int32_t>>& polys, const std::vector<double>& holes,
                       std::vector<double>& P, std::vector<int32_t>& tris, const std::string& where, bool& flat,
                       double h = 0) {
    flat = false;

    if (polys.size() == 1 && holes.empty() && polys[0].size() == 3 && h <= 0) {
        const auto& p = polys[0];

        if (p[0] != p[1] && p[1] != p[2] && p[0] != p[2]) {
            tris.insert(tris.end(), p.begin(), p.end());
        } else {
            flat = true;   // (a corner repeated: a segment, as iso2mesh writes one)
        }

        return;
    }

    // the plane: Newell's normal of the polygons
    double nrm[3] = { 0, 0, 0 };

    for (const auto& p : polys)
        if (p.size() >= 3)
            for (size_t i = 0; i < p.size(); ++i) {
                const double* a = &P[3 * static_cast<size_t>(p[i])], *b = &P[3 * static_cast<size_t>(p[(i + 1) % p.size()])];
                nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
                nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
                nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
            }

    const int drop = std::fabs(nrm[0]) >= std::fabs(nrm[1]) && std::fabs(nrm[0]) >= std::fabs(nrm[2]) ? 0 :
                     std::fabs(nrm[1]) >= std::fabs(nrm[2]) ? 1 : 2;
    const int ax = (drop + 1) % 3, ay = (drop + 2) % 3;

    if (!(std::fabs(nrm[drop]) > 0)) {
        flat = true;   // no area: segments / points only
        return;
    }

    // the points (a row once; two rows at one 2-D point merged)
    FacetCDT C;
    C.where = where;
    std::unordered_map<int32_t, int> loc;
    std::map<std::pair<double, double>, int> at2;
    std::vector<int32_t> glob;
    std::vector<double> X;

    for (const auto& p : polys)
        for (const int32_t r : p) {
            if (loc.count(r)) {
                continue;
            }

            const std::pair<double, double> xy{ P[3 * static_cast<size_t>(r) + static_cast<size_t>(ax)],
                                                P[3 * static_cast<size_t>(r) + static_cast<size_t>(ay)] };
            const auto it = at2.find(xy);

            if (it != at2.end()) {
                loc[r] = it->second;
                continue;
            }

            const int l = static_cast<int>(glob.size());
            at2[xy] = l;
            loc[r] = l;
            glob.push_back(r);
            X.push_back(xy.first);
            X.push_back(xy.second);
        }

    if (glob.size() < 3) {
        flat = true;
        return;
    }

    // interior points (h > 0): a grid inside the polygons (even-odd), half a
    // spacing clear of their edges; placeholders -1 - k, made nodes when used
    std::vector<std::array<double, 2>> grid;

    // (only on a facet in an axis plane -- its corners all at one coordinate -- where a
    // point is exactly on it: on a slanted one a computed point misses the plane by a
    // rounding, the exact CDT sees a fold and fills it with flat slivers; there the
    // edges' points alone make the triangles)
    bool axial = h > 0;

    for (const auto& p : polys)
        for (const int32_t r : p) {
            axial = axial && P[3 * static_cast<size_t>(r) + static_cast<size_t>(drop)] ==
                    P[3 * static_cast<size_t>(polys[0][0]) + static_cast<size_t>(drop)];
        }

    if (axial) {
        const double nn = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        const double g = h * std::fabs(nrm[drop]) / nn;   // (the projection shrinks the tilted way)
        double lo[2] = { 1e300, 1e300 }, hi[2] = { -1e300, -1e300 };

        for (size_t i = 0; i < X.size(); i += 2)
            for (int a = 0; a < 2; ++a) {
                lo[a] = std::min(lo[a], X[i + static_cast<size_t>(a)]);
                hi[a] = std::max(hi[a], X[i + static_cast<size_t>(a)]);
            }

        std::vector<std::array<int, 2>> ed;   // the polygons' edges (local points)

        for (const auto& p : polys)
            if (p.size() >= 3)
                for (size_t i = 0; i < p.size(); ++i) {
                    ed.push_back({ { loc[p[i]], loc[p[(i + 1) % p.size()]] } });
                }

        const int nx = static_cast<int>(std::floor((hi[0] - lo[0]) / g)), ny = static_cast<int>(std::floor((hi[1] - lo[1]) / g));

        if (nx >= 1 && ny >= 1 && static_cast<double>(nx) * ny < 4e6)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    const double x = lo[0] + (hi[0] - lo[0] - nx * g) * 0.5 + i * g - 0.5 * g * (j & 1);
                    const double y = lo[1] + (hi[1] - lo[1] - ny * g) * 0.5 + j * g - 0.5 * g;
                    bool in = false, near = false;

                    for (const auto& e : ed) {
                        const double ax2 = X[2 * static_cast<size_t>(e[0])], ay2 = X[2 * static_cast<size_t>(e[0]) + 1];
                        const double bx = X[2 * static_cast<size_t>(e[1])], by = X[2 * static_cast<size_t>(e[1]) + 1];

                        if ((ay2 > y) != (by > y) && x < (bx - ax2) * (y - ay2) / (by - ay2) + ax2) {
                            in = !in;
                        }

                        const double ex = bx - ax2, ey = by - ay2, l2 = ex * ex + ey * ey;
                        const double t = l2 > 0 ? std::max(0.0, std::min(1.0, ((x - ax2) * ex + (y - ay2) * ey) / l2)) : 0.0;
                        const double dx = x - (ax2 + t * ex), dy = y - (ay2 + t * ey);
                        near = near || dx * dx + dy * dy < 0.25 * g * g;
                    }

                    if (in && !near) {
                        glob.push_back(-1 - static_cast<int32_t>(grid.size()));
                        grid.push_back({ { x, y } });
                        X.push_back(x);
                        X.push_back(y);
                    }
                }
    }

    C.n = static_cast<int>(glob.size());
    delaunay2d(X, C.T, true);
    C.X = X;
    const int nt = static_cast<int>(C.T.size() / 3);

    for (int t = 0; t < nt; ++t) {
        C.link(t);
    }

    // the polygon edges
    for (const auto& p : polys) {
        if (p.size() == 2) {
            C.insert(loc[p[0]], loc[p[1]]);
        } else if (p.size() >= 3)
            for (size_t i = 0; i < p.size(); ++i) {
                C.insert(loc[p[i]], loc[p[(i + 1) % p.size()]]);
            }
    }

    // outside: from the enclosing triangle; the holes: from the triangle holding each
    std::vector<char> out(static_cast<size_t>(nt), 0);

    for (int t = 0; t < nt; ++t)
        if (!out[static_cast<size_t>(t)] && (C.T[3 * t] >= C.n || C.T[3 * t + 1] >= C.n || C.T[3 * t + 2] >= C.n)) {
            C.flood(t, out);
        }

    for (size_t h = 0; h + 2 < holes.size(); h += 3) {
        const size_t hp = C.X.size() / 2;
        C.X.push_back(holes[h + static_cast<size_t>(ax)]);
        C.X.push_back(holes[h + static_cast<size_t>(ay)]);

        for (int t = 0; t < nt; ++t) {
            const int* v = &C.T[3 * t];

            if (C.o2(v[0], v[1], static_cast<int>(hp)) >= 0 && C.o2(v[1], v[2], static_cast<int>(hp)) >= 0 &&
                    C.o2(v[2], v[0], static_cast<int>(hp)) >= 0) {
                if (!out[static_cast<size_t>(t)]) {
                    C.flood(t, out);
                }

                break;
            }
        }

        C.X.resize(2 * hp);
    }

    // (the grid's points: on the facet's plane)
    const size_t r0 = 3 * static_cast<size_t>(polys[0][0]);
    std::vector<int32_t> made(grid.size(), -1);

    for (int t = 0; t < nt; ++t)
        if (!out[static_cast<size_t>(t)]) {
            for (int k = 0; k < 3; ++k) {
                int32_t id = glob[static_cast<size_t>(C.T[3 * t + k])];

                if (id < 0) {
                    const size_t gk = static_cast<size_t>(-1 - id);

                    if (made[gk] < 0) {
                        double q[3];
                        q[ax] = grid[gk][0];
                        q[ay] = grid[gk][1];
                        q[drop] = P[r0 + static_cast<size_t>(drop)];   // (an axis plane: exact)
                        made[gk] = static_cast<int32_t>(P.size() / 3);
                        P.insert(P.end(), q, q + 3);
                    }

                    id = made[gk];
                }

                tris.push_back(id);
            }
        }
}

// The facets' polygon edges split where another point lies on them (so two
// facets sharing an edge share its points too), and, h > 0, into pieces no
// longer than h -- each edge once, its new points shared by all its facets
void refine_edges(std::vector<std::vector<std::vector<int32_t>>>& facets, std::vector<double>& P, double h) {
    const size_t n0 = P.size() / 3;
    auto pt = [&](int32_t i) {
        return &P[3 * static_cast<size_t>(i)];
    };
    // the points hashed, cells of the mean edge length
    double esum = 0;
    size_t ecount = 0;

    for (const auto& f : facets)
        for (const auto& p : f)
            for (size_t i = 0; p.size() >= 2 && i < p.size(); ++i) {
                const double* a = pt(p[i]), *b = pt(p[(i + 1) % p.size()]);
                esum += std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
                ++ecount;
            }

    const double cell = ecount ? std::max(1e-12, esum / static_cast<double>(ecount)) : 1.0;
    auto ck = [&](int64_t i, int64_t j, int64_t k) {
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };
    std::unordered_map<int64_t, std::vector<int32_t>> bins;

    for (size_t i = 0; i < n0; ++i) {
        const double* a = pt(static_cast<int32_t>(i));
        bins[ck(static_cast<int64_t>(std::floor(a[0] / cell)), static_cast<int64_t>(std::floor(a[1] / cell)),
                static_cast<int64_t>(std::floor(a[2] / cell)))].push_back(static_cast<int32_t>(i));
    }

    std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> done;   // (lo, hi) -> its inner points, lo to hi
    auto chain = [&](int32_t a, int32_t b) {   // the inner points of edge a -> b, in order
        const bool flip = a > b;
        const std::pair<int32_t, int32_t> key(std::min(a, b), std::max(a, b));
        auto it = done.find(key);

        if (it == done.end()) {
            const double* A = pt(key.first), *B = pt(key.second);
            const double d[3] = { B[0] - A[0], B[1] - A[1], B[2] - A[2] };
            const double L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], L = std::sqrt(L2);
            std::vector<std::pair<double, int32_t>> on;   // (parameter, point) of the points on it
            int64_t lo[3], hi[3];

            for (int k = 0; k < 3; ++k) {
                lo[k] = static_cast<int64_t>(std::floor(std::min(A[k], B[k]) / cell));
                hi[k] = static_cast<int64_t>(std::floor(std::max(A[k], B[k]) / cell));
            }

            if ((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) <= 100000)
                for (int64_t x = lo[0]; x <= hi[0]; ++x)
                    for (int64_t y = lo[1]; y <= hi[1]; ++y)
                        for (int64_t z = lo[2]; z <= hi[2]; ++z) {
                            const auto bt = bins.find(ck(x, y, z));

                            if (bt == bins.end()) {
                                continue;
                            }

                            for (const int32_t c : bt->second) {
                                if (c == key.first || c == key.second || !(L2 > 0)) {
                                    continue;
                                }

                                const double* Q = pt(c);
                                const double q[3] = { Q[0] - A[0], Q[1] - A[1], Q[2] - A[2] };
                                const double t = (q[0] * d[0] + q[1] * d[1] + q[2] * d[2]) / L2;

                                if (t <= 1e-9 || t >= 1 - 1e-9) {
                                    continue;
                                }

                                const double r[3] = { q[0] - t * d[0], q[1] - t * d[1], q[2] - t * d[2] };

                                if (std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) <= 1e-9 * std::max(L, 1.0)) {
                                    on.emplace_back(t, c);
                                }
                            }
                        }

            std::sort(on.begin(), on.end());
            on.erase(std::unique(on.begin(), on.end(), [](const std::pair<double, int32_t>& x,
            const std::pair<double, int32_t>& y) {
                return x.second == y.second;
            }), on.end());
            // the pieces between them, each split to h
            std::vector<int32_t> inner;
            std::vector<std::pair<double, int32_t>> ends;
            ends.emplace_back(0.0, key.first);
            ends.insert(ends.end(), on.begin(), on.end());
            ends.emplace_back(1.0, key.second);

            for (size_t k = 0; k + 1 < ends.size(); ++k) {
                if (k > 0) {
                    inner.push_back(ends[k].second);
                }

                const int m = h > 0 ? static_cast<int>(std::ceil((ends[k + 1].first - ends[k].first) * L / h)) : 1;

                for (int s = 1; s < m; ++s) {
                    const double t = ends[k].first + (ends[k + 1].first - ends[k].first) * s / m;
                    const double q[3] = { A[0] + t * d[0], A[1] + t * d[1], A[2] + t * d[2] };
                    inner.push_back(static_cast<int32_t>(P.size() / 3));
                    P.insert(P.end(), q, q + 3);
                    A = pt(key.first);   // (P may have moved)
                    B = pt(key.second);
                }
            }

            it = done.emplace(key, inner).first;
        }

        std::vector<int32_t> r = it->second;

        if (flip) {
            std::reverse(r.begin(), r.end());
        }

        return r;
    };

    for (auto& f : facets)
        for (auto& p : f) {
            if (p.size() < 2) {
                continue;
            }

            std::vector<int32_t> q;
            const size_t m = p.size() == 2 ? 1 : p.size();   // (a segment: its one edge)

            for (size_t i = 0; i < m; ++i) {
                q.push_back(p[i]);
                const std::vector<int32_t> in = chain(p[i], p[(i + 1) % p.size()]);
                q.insert(q.end(), in.begin(), in.end());
            }

            if (p.size() == 2) {
                q.push_back(p[1]);
            }

            p = q;
        }
}

}  // namespace

void cdt2d(const std::vector<double>& P, const std::vector<int>& segs, std::vector<int>& tris, std::vector<int>& comp,
           std::vector<char>& comp_out) {
    FacetCDT C;
    C.where = "cdt2d";
    std::vector<double> X(P);
    C.n = static_cast<int>(P.size() / 2);
    tris.clear();
    comp.clear();
    comp_out.clear();

    if (C.n < 3) {
        return;
    }

    delaunay2d(X, C.T, true);
    C.X = X;
    const int nt = static_cast<int>(C.T.size() / 3);

    for (int t = 0; t < nt; ++t) {
        C.link(t);
    }

    for (size_t k = 0; k + 1 < segs.size(); k += 2) {
        C.insert(segs[k], segs[k + 1]);
    }

    // the components across the non-constraint edges; one holding a corner of
    // the enclosing triangle is outside
    std::vector<int> cid(static_cast<size_t>(nt), -1);
    std::vector<char> outc;

    for (int s0 = 0; s0 < nt; ++s0) {
        if (cid[static_cast<size_t>(s0)] >= 0) {
            continue;
        }

        std::vector<char> in(static_cast<size_t>(nt), 0);
        C.flood(s0, in);
        const int c = static_cast<int>(outc.size());
        char o = 0;

        for (int t = 0; t < nt; ++t)
            if (in[static_cast<size_t>(t)]) {
                cid[static_cast<size_t>(t)] = c;
                o = o || C.T[3 * t] >= C.n || C.T[3 * t + 1] >= C.n || C.T[3 * t + 2] >= C.n;
            }

        outc.push_back(o);
    }

    for (int t = 0; t < nt; ++t)
        if (C.T[3 * t] < C.n && C.T[3 * t + 1] < C.n && C.T[3 * t + 2] < C.n) {
            tris.insert(tris.end(), { C.T[3 * t], C.T[3 * t + 1], C.T[3 * t + 2] });
            comp.push_back(cid[static_cast<size_t>(t)]);
        }

    comp_out = outc;
}

Mesh read_poly(const std::string& path, PlcStats* st, double h) {
    PlcStats S;
    Lines in(path);
    std::string ext = path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const bool smesh = ext == ".smesh";
    Mesh m;
    std::unordered_map<long, int32_t> row;
    long base = 0;

    // part 1: the points (none: from the .node file beside it)
    {
        const size_t h = in.at;
        const long n = to_long(in.next("the node list")[0], path);

        if (n == 0) {   // (as TetGen: the .node file of the same name, beside it)
            std::string np = path.substr(0, path.size() - ext.size()) + ".node";

            if (!std::ifstream(np)) {
                // else one the line's comment names ("nodes are found in file x.node"), beside it
                std::istringstream cs(in.C[h]);
                std::string w, alt;
                const size_t sl = path.find_last_of("/\\");
                const std::string dir = sl == std::string::npos ? std::string() : path.substr(0, sl + 1);

                while (cs >> w)
                    if (w.size() > 5 && w.compare(w.size() - 5, 5, ".node") == 0) {
                        alt = dir + w;
                        break;
                    }

                if (alt.empty() || !std::ifstream(alt)) {
                    throw std::runtime_error(path + ": it lists no points (0 in its node section): they belong in " + np +
                                             (alt.empty() ? std::string() : " (or " + alt + ")") + ", which is not there");
                }

                np = alt;
            }

            Lines nd(np);
            read_nodes(nd, m.nodes, row, base);
        } else {
            in.at = h;
            read_nodes(in, m.nodes, row, base);
        }
    }

    S.points = m.nodes.size() / 3;
    auto node = [&](const std::string & s) {
        const auto it = row.find(to_long(s, path));

        if (it == row.end()) {
            throw std::runtime_error(path + ": a facet refers to a point not listed: " + s);
        }

        return it->second;
    };
    // part 2: the facets
    const auto& fh = in.next("the facet list");
    const long nf = to_long(fh[0], path);
    const bool fmark = fh.size() > 1 && to_long(fh[1], path) != 0;
    (void)fmark;

    std::vector<std::vector<std::vector<int32_t>>> fpolys;   // (all facets first: their edges are refined together)
    std::vector<std::vector<double>> fhole;
    std::vector<std::string> fwhere;

    for (long f = 0; f < nf; ++f) {
        std::vector<std::vector<int32_t>> polys;
        std::vector<double> fholes;
        const std::string where = path + ": facet " + std::to_string(f + base);

        if (smesh) {   // one polygon per line: <n> <corners..> [marker]
            const std::vector<std::string> t = in.counted("the facets");
            const long c = to_long(t[0], path);

            if (static_cast<long>(t.size()) < 1 + c) {
                throw std::runtime_error(where + ": fewer corners than it says");
            }

            std::vector<int32_t> p;

            for (long j = 0; j < c; ++j) {
                p.push_back(node(t[static_cast<size_t>(1 + j)]));
            }

            polys.push_back(p);
        } else {   // <polygons> [holes] [marker], the polygons, the hole points
            const auto& t = in.next("the facets");
            const long np = to_long(t[0], path);
            const long nh = t.size() > 1 ? to_long(t[1], path) : 0;

            for (long k = 0; k < np; ++k) {
                const std::vector<std::string> q = in.counted("a facet's polygons");
                const long c = to_long(q[0], path);

                if (static_cast<long>(q.size()) < 1 + c) {
                    throw std::runtime_error(where + ": a polygon with fewer corners than it says");
                }

                std::vector<int32_t> p;

                for (long j = 0; j < c; ++j) {
                    p.push_back(node(q[static_cast<size_t>(1 + j)]));
                }

                polys.push_back(p);
            }

            for (long k = 0; k < nh; ++k) {
                const auto& q = in.next("a facet's holes");

                if (q.size() < 4) {
                    throw std::runtime_error(where + ": a hole line needs an index and x y z");
                }

                for (int a = 0; a < 3; ++a) {
                    fholes.push_back(to_double(q[static_cast<size_t>(1 + a)], path));
                }
            }

            S.facet_holes += static_cast<size_t>(nh);
        }

        S.polygons += polys.size();
        fpolys.push_back(polys);
        fhole.push_back(fholes);
        fwhere.push_back(where);
    }

    // the edges refined (shared: the facets' points match), then each facet
    refine_edges(fpolys, m.nodes, h);

    for (size_t f = 0; f < fpolys.size(); ++f) {
        bool flat = false;
        triangulate_facet(fpolys[f], fhole[f], m.nodes, m.tris, fwhere[f], flat, h);
        S.flat_facets += flat;
    }

    S.facets = static_cast<size_t>(nf);

    // part 3: the volume holes; part 4 (optional): the regions
    if (in.more()) {
        const long nh = to_long(in.next("the hole list")[0], path);

        for (long k = 0; k < nh; ++k) {
            const auto& q = in.next("the holes");

            if (q.size() < 4) {
                throw std::runtime_error(path + ": a hole line needs an index and x y z");
            }

            for (int a = 0; a < 3; ++a) {
                m.seed_holes.push_back(to_double(q[static_cast<size_t>(1 + a)], path));
            }
        }
    }

    if (in.more()) {
        const long nr = to_long(in.next("the region list")[0], path);

        for (long k = 0; k < nr; ++k) {
            const auto& q = in.next("the regions");

            if (q.size() < 5) {
                throw std::runtime_error(path + ": a region line needs an index, x y z and a region number");
            }

            for (int a = 0; a < 4; ++a) {
                m.seed_regions.push_back(to_double(q[static_cast<size_t>(1 + a)], path));
            }
        }
    }

    S.triangles = m.tris.size() / 3;
    S.holes = m.seed_holes.size() / 3;
    S.regions = m.seed_regions.size() / 4;

    if (st) {
        *st = S;
    }

    return m;
}

}  // namespace tn
