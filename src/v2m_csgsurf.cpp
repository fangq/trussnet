// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_csgsurf.cpp -- see v2m_csgsurf.h.

#include "v2m_csgsurf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "v2m_modes.h"      // compact_nodes
#include "v2m_plc.h"        // cdt2d
#include "v2m_surfgeom.h"

namespace tn {

namespace csg_host {

using std::fabs;
using std::fmax;
using std::fmin;
using std::sqrt;
#define V2M_G
#include "opencl/v2m_sdf_body.cl"
#undef V2M_G

}  // namespace csg_host

using namespace csg_host;

namespace {

using namespace sg;

// an edge of the patches: surface S along the straight (u, v) line a -> b, t in
// [0, 1]; its nodes (by t, ascending) shared by every patch it bounds
struct Edge {
    Surf S;
    double ua = 0, va = 0, ub = 0, vb = 0;
    bool degen = false;   // a pole / an apex / a disk's centre: one node
    std::vector<double> t;
    std::vector<int32_t> n;
    A3 at(double tt) const {
        return S.eval(ua + tt * (ub - ua), va + tt * (vb - va));
    }
};

// a patch: [u0, u1] x [v0, v1] of S; sides 0 bottom (v0, u0 -> u1), 1 right (u1,
// v0 -> v1), 2 top (v1, u0 -> u1), 3 left (u0, v0 -> v1), each an edge (rev: the
// edge's t runs the other way)
struct Patch {
    Surf S;
    double u0 = 0, u1 = 1, v0 = 0, v1 = 1;
    int prim = 0;
    int e[4] = { -1, -1, -1, -1 };
    bool rev[4] = { false, false, false, false };
    std::array<double, 6> box{ { 0, 0, 0, 0, 0, 0 } };
    void side_uv(int s, double sg2, double& u, double& v) const {
        switch (s) {
            case 0:
                u = u0 + sg2 * (u1 - u0);
                v = v0;
                break;

            case 1:
                u = u1;
                v = v0 + sg2 * (v1 - v0);
                break;

            case 2:
                u = u0 + sg2 * (u1 - u0);
                v = v1;
                break;

            default:
                u = u0;
                v = v0 + sg2 * (v1 - v0);
        }
    }
};

// where two primitives' surfaces cross: marched on a patch of primitive a, the
// other b; its nodes in order (shared by the patches of both)
struct Chain {
    int a = 0, b = 0;
    std::vector<int32_t> n;
};

struct Csg {
    const ShapeScene& sc;
    CsgSurfOptions o;
    std::vector<double> X;
    std::vector<Edge> E;
    std::vector<Patch> Pt;
    std::vector<Chain> C;
    std::vector<float> lprog;   // the labels' fields (exact)
    int np = 0, nl = 0;
    double diag = 1, tol = 0, hmax = 0, amax = 0, near = 0;
    CsgSurfStats st;

    explicit Csg(const ShapeScene& s, const CsgSurfOptions& op) : sc(s), o(op) {}

    int32_t add(const A3& p) {
        X.insert(X.end(), p.begin(), p.end());
        return static_cast<int32_t>(X.size() / 3 - 1);
    }
    A3 at(int32_t i) const {
        return { { X[3 * static_cast<size_t>(i)], X[3 * static_cast<size_t>(i) + 1], X[3 * static_cast<size_t>(i) + 2] } };
    }
    // each primitive's field in double precision: its zero set exactly its
    // patches' surfaces (the crossings' roots on both; s > 0 inside)
    struct Field {
        int type = 0;
        A3 a{ { 0, 0, 0 } }, b{ { 0, 0, 0 } }, z{ { 0, 0, 1 } };
        double r = 0, r2 = 0, L = 0;
    };
    std::vector<Field> PF;
    double fprim(int q, const A3& p) const {
        const Field& F = PF[static_cast<size_t>(q)];

        switch (F.type) {
            case V2M_SDF_BOX: {
                double d = -1e300;

                for (size_t k = 0; k < 3; ++k) {
                    d = std::max(d, std::fabs(p[k] - 0.5 * (F.a[k] + F.b[k])) - 0.5 * (F.b[k] - F.a[k]));
                }

                return -d;
            }

            case V2M_SDF_SPHERE:
                return F.r - norm(p - F.a);

            case V2M_SDF_TORUS: {
                const A3 d = p - F.a;
                const double h = dot(d, F.z), rho = norm(d - h * F.z);
                return F.r2 - std::hypot(rho - F.r, h);
            }

            default: {   // a cylinder, a cone: the side and the two caps
                const A3 d = p - F.a;
                const double h = dot(d, F.z), rho = norm(d - h * F.z);
                const double k = (F.r2 - F.r) / F.L, side = (F.r + h * k - rho) / std::sqrt(1 + k * k);
                return std::min(side, std::min(h, F.L - h));
            }
        }
    }
    int label(const A3& p) const {
        int best = 0;
        float bv = -1e30f;

        for (int l = 0; l < nl; ++l) {
            const float v = v2m_sdf_eval(lprog.data(), l, static_cast<float>(p[0]), static_cast<float>(p[1]),
                                         static_cast<float>(p[2]), nullptr);

            if (v > bv) {
                bv = v;
                best = l;
            }
        }

        return best;
    }

    // ---- edges: sampled once (the chord tolerance, the turn, --size) -----------------
    void sample(const Edge& e, double a, double b, const A3& pa, const A3& pb, int depth, std::vector<double>& T) {
        const double m = 0.5 * (a + b);
        const A3 pm = e.at(m), ab = pb - pa;
        const double L = norm(ab);
        const double dev = L > 0 ? norm(cross(pm - pa, ab)) / L : norm(pm - pa);
        const A3 h1 = pm - pa, h2 = pb - pm;
        const double c12 = norm(h1) > 0 && norm(h2) > 0 ? dot(h1, h2) / (norm(h1) * norm(h2)) : 1;

        if (depth < 16 && (depth < 1 || dev > tol || c12 < std::cos(amax) || (hmax > 0 && L > hmax))) {
            sample(e, a, m, pa, pm, depth + 1, T);
            sample(e, m, b, pm, pb, depth + 1, T);
        } else {
            T.push_back(b);
        }
    }
    int edge(const Surf& S, double ua, double va, double ub, double vb, int32_t na, int32_t nb, bool degen = false) {
        Edge e;
        e.S = S;
        e.ua = ua;
        e.va = va;
        e.ub = ub;
        e.vb = vb;
        e.degen = degen;

        if (degen) {
            e.t = { 0.0 };
            e.n = { na };
        } else {
            std::vector<double> T{ 0.0 };
            sample(e, 0, 1, at(na), at(nb), 0, T);
            e.t.push_back(0);
            e.n.push_back(na);

            for (size_t k = 1; k + 1 < T.size(); ++k) {
                e.t.push_back(T[k]);
                e.n.push_back(add(e.at(T[k])));
            }

            e.t.push_back(1);
            e.n.push_back(nb);
        }

        E.push_back(e);
        return static_cast<int>(E.size()) - 1;
    }
    // a node on edge ei at t: an existing one if close, else a new one there
    int32_t on_edge(int ei, double t, const A3& p) {
        Edge& e = E[static_cast<size_t>(ei)];

        if (e.degen) {
            return e.n[0];
        }

        for (size_t k = 0; k < e.n.size(); ++k)
            if (norm(at(e.n[k]) - p) <= near) {
                return e.n[k];
            }

        const size_t k = static_cast<size_t>(std::upper_bound(e.t.begin(), e.t.end(), t) - e.t.begin());
        const int32_t id = add(p);
        e.t.insert(e.t.begin() + static_cast<std::ptrdiff_t>(k), t);
        e.n.insert(e.n.begin() + static_cast<std::ptrdiff_t>(k), id);
        return id;
    }

    // ---- the primitives' patches -------------------------------------------------------
    void patch(const Surf& S, double u0, double u1, double v0, double v1, int prim, const int e[4], const bool rev[4]) {
        Patch p;
        p.S = S;
        p.u0 = u0;
        p.u1 = u1;
        p.v0 = v0;
        p.v1 = v1;
        p.prim = prim;

        for (int k = 0; k < 4; ++k) {
            p.e[k] = e[k];
            p.rev[k] = rev[k];
        }

        A3 lo{ { 1e300, 1e300, 1e300 } }, hi{ { -1e300, -1e300, -1e300 } };

        for (int i = 0; i <= 8; ++i)
            for (int j = 0; j <= 8; ++j) {
                const A3 q = S.eval(u0 + (u1 - u0) * i / 8, v0 + (v1 - v0) * j / 8);

                for (int a = 0; a < 3; ++a) {
                    lo[static_cast<size_t>(a)] = std::min(lo[static_cast<size_t>(a)], q[static_cast<size_t>(a)]);
                    hi[static_cast<size_t>(a)] = std::max(hi[static_cast<size_t>(a)], q[static_cast<size_t>(a)]);
                }
            }

        const double pad = 0.05 * norm(hi - lo) + 1e-6 * diag;
        p.box = { { lo[0] - pad, lo[1] - pad, lo[2] - pad, hi[0] + pad, hi[1] + pad, hi[2] + pad } };
        Pt.push_back(p);
    }
    static Frame frame_z(const A3& o, const A3& z) {
        Frame f;
        f.o = o;
        f.z = unit(z);
const A3 a = std::fabs(f.z[0]) < 0.9 ? A3{ { 1, 0, 0 } } :
        A3{ { 0, 1, 0 } };
        f.x = unit(cross(a, f.z));
        f.y = cross(f.z, f.x);
        return f;
    }
    void box(int prim, const A3& lo, const A3& hi) {
        A3 c[8];

        for (int k = 0; k < 8; ++k) {
            c[k] = { { k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2] } };
        }

        int32_t n[8];

        for (int k = 0; k < 8; ++k) {
            n[k] = add(c[k]);
        }

        std::map<std::pair<int, int>, int> ed;   // (corner, corner) -> the edge, from the first to the second
        auto get = [&](int a, int b, bool & rev) {
            const auto it = ed.find({ b, a }), it2 = ed.find({ a, b });

            if (it != ed.end()) {
                rev = true;
                return it->second;
            }

            if (it2 != ed.end()) {
                rev = false;
                return it2->second;
            }

            rev = false;
            Surf S;
            S.k = Surf::PLANE;
            S.f.o = c[a];
            const A3 d = c[b] - c[a];
            S.f.x = unit(d);
            S.f.y = unit(cross(std::fabs(S.f.x[0]) < 0.9 ? A3{ { 1, 0, 0 } } : A3{ { 0, 1, 0 } }, S.f.x));
            const int id = edge(S, 0, 0, norm(d), 0, n[a], n[b]);
            ed[ { a, b }] = id;
            return id;
        };
        // the faces: corners c0 c1 c2 c3 (u along c0 -> c1, v along c0 -> c3)
        static const int F[6][4] = { { 0, 1, 3, 2 }, { 4, 5, 7, 6 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 2, 6, 4 }, { 1, 3, 7, 5 } };

        for (const auto& f : F) {
            // (as c0 c1 c2 c3 of a rectangle: c0 -> c1 -> c2' -> c3, c2' = c1 + c3 - c0)
            const int q0 = f[0], q1 = f[1], q3 = f[3], q2 = f[2];
            Surf S;
            S.k = Surf::PLANE;
            S.f.o = c[q0];
            S.f.x = unit(c[q1] - c[q0]);
            S.f.y = unit(c[q3] - c[q0]);
            const double Lu = norm(c[q1] - c[q0]), Lv = norm(c[q3] - c[q0]);
            int e[4];
            bool rv[4];
            e[0] = get(q0, q1, rv[0]);   // bottom c0 -> c1
            e[1] = get(q1, q2, rv[1]);   // right c1 -> c2
            e[2] = get(q3, q2, rv[2]);   // top c3 -> c2
            e[3] = get(q0, q3, rv[3]);   // left c0 -> c3
            patch(S, 0, Lu, 0, Lv, prim, e, rv);
        }
    }
    // a surface of revolution about f.z, u the angle in two halves [0, pi] [pi, 2 pi],
    // v in [va, vb]; ends: a rim (a circle, shared with a cap) or a pole
    void revolved(int prim, const Surf& S, double va, double vb, bool pole_a, bool pole_b, int rimA[2], int rimB[2],
                  int32_t& seamA0, int32_t& seamB0, int32_t& seamApi, int32_t& seamBpi) {
        const int32_t pa = pole_a ? add(S.eval(0, va)) : -1, pb = pole_b ? add(S.eval(0, vb)) : -1;
        seamA0 = pole_a ? pa : add(S.eval(0, va));
        seamApi = pole_a ? pa : add(S.eval(kPi, va));
        seamB0 = pole_b ? pb : add(S.eval(0, vb));
        seamBpi = pole_b ? pb : add(S.eval(kPi, vb));
        const int s0 = edge(S, 0, va, 0, vb, seamA0, seamB0);        // the seams, va -> vb
        const int s1 = edge(S, kPi, va, kPi, vb, seamApi, seamBpi);

        for (int h = 0; h < 2; ++h) {
            const double u0 = h * kPi, u1 = u0 + kPi;
            const int32_t na0 = h ? seamApi : seamA0, na1 = h ? seamA0 : seamApi;
            const int32_t nb0 = h ? seamBpi : seamB0, nb1 = h ? seamB0 : seamBpi;
            rimA[h] = pole_a ? edge(S, u0, va, u1, va, pa, pa, true) : edge(S, u0, va, u1, va, na0, na1);
            rimB[h] = pole_b ? edge(S, u0, vb, u1, vb, pb, pb, true) : edge(S, u0, vb, u1, vb, nb0, nb1);
            const int e[4] = { rimA[h], h ? s0 : s1, rimB[h], h ? s1 : s0 };
            const bool rv[4] = { false, false, false, false };
            patch(S, u0, u1, va, vb, prim, e, rv);
        }
    }
    // a disk of radius r at centre c (the frame of the side it caps), its rim halves
    void cap(int prim, const Frame& f, double r, const int rim[2], int32_t r0, int32_t rpi) {
        Surf D;
        D.k = Surf::DISK;
        D.f = f;
        D.stol = 1e-7 * diag;
        const int32_t c = add(f.o);
        const int d0 = edge(D, 0, 0, 0, r, c, r0), d1 = edge(D, kPi, 0, kPi, r, c, rpi);

        for (int h = 0; h < 2; ++h) {
            const double u0 = h * kPi, u1 = u0 + kPi;
            const int ctr = edge(D, u0, 0, u1, 0, c, c, true);
            const int e[4] = { ctr, h ? d0 : d1, rim[h], h ? d1 : d0 };
            const bool rv[4] = { false, false, false, false };
            patch(D, u0, u1, 0, r, prim, e, rv);
        }
    }
    void build_primitive(int q) {
        const std::vector<float>& c = sc.pcode[static_cast<size_t>(q)];

        if (c.size() < 2 || static_cast<int>(c[0]) != V2M_SDF_PRIM) {
            throw std::runtime_error("exact surface: a primitive not understood");
        }

        const int type = static_cast<int>(c[1]);
        const float* p = &c[2];

        if (type == V2M_SDF_BOX) {
            box(q, { { p[0], p[1], p[2] } }, { { p[3], p[4], p[5] } });
        } else if (type == V2M_SDF_SPHERE) {
            Surf S;
            S.k = Surf::SPHERE;
            S.f = frame_z({ { p[0], p[1], p[2] } }, { { 0, 0, 1 } });
            S.r = p[3];
            S.stol = 1e-7 * diag;
            int ra[2], rb[2];
            int32_t a0, b0, api, bpi;
            revolved(q, S, -0.5 * kPi, 0.5 * kPi, true, true, ra, rb, a0, b0, api, bpi);
        } else if (type == V2M_SDF_CYL || type == V2M_SDF_CONE) {
            const A3 a{ { p[0], p[1], p[2] } }, b{ { p[3], p[4], p[5] } };
            const double ra = p[6], rb = type == V2M_SDF_CONE ? p[7] : p[6], L = norm(b - a);
            Surf S;
            S.f = frame_z(a, b - a);
            S.stol = 1e-7 * diag;

            if (type == V2M_SDF_CYL || std::fabs(rb - ra) < 1e-12 * L) {
                S.k = Surf::CYL;
                S.r = ra;
            } else {
                S.k = Surf::CONE;
                S.r = ra;
                S.ang = std::atan((rb - ra) / L);
            }

            int rA[2], rB[2];
            int32_t a0, b0, api, bpi;
            revolved(q, S, 0, L, ra <= 0, rb <= 0, rA, rB, a0, b0, api, bpi);

            if (ra > 0) {
                cap(q, S.f, ra, rA, a0, api);
            }

            if (rb > 0) {
                Frame fb = S.f;
                fb.o = b;
                cap(q, fb, rb, rB, b0, bpi);
            }
        } else if (type == V2M_SDF_TORUS) {
            Surf S;
            S.k = Surf::TORUS;
            S.f = frame_z({ { p[0], p[1], p[2] } }, { { p[3], p[4], p[5] } });
            S.R = p[6];
            S.r = p[7];
            S.stol = 1e-7 * diag;
            // 4 quarters: u, v in [0, pi], [pi, 2 pi]; the 4 seam circles shared
            int32_t nd[2][2];

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    nd[i][j] = add(S.eval(i * kPi, j * kPi));
                }

            int U[2][2], V[2][2];   // U[j][i]: v = j pi, u in [i pi, (i+1) pi]; V[i][j]: u = i pi, v in [j pi, (j+1) pi]

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    U[j][i] = edge(S, i * kPi, j * kPi, (i + 1) * kPi, j * kPi, nd[i][j], nd[(i + 1) % 2][j]);
                    V[i][j] = edge(S, i * kPi, j * kPi, i * kPi, (j + 1) * kPi, nd[i][j], nd[i][(j + 1) % 2]);
                }

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    const int e[4] = { U[j][i], V[(i + 1) % 2][j], U[(j + 1) % 2][i], V[i][j] };
                    const bool rv[4] = { false, false, false, false };
                    patch(S, i * kPi, (i + 1) * kPi, j * kPi, (j + 1) * kPi, q, e, rv);
                }
        } else {
            throw std::runtime_error("exact surface: planes, slabs, layers, ellipsoids and lenses are not supported yet "
                                     "(use --mode mesh)");
        }
    }

    // ---- a point's (u, v) on patch G (false: not on it) ---------------------------------
    bool locate(const Patch& G, const A3& p, double& u, double& v) const {
        bool sing;
        G.S.inv(p, u, v, sing, 0, 0, false);
        const double eu = 1e-9 * (G.u1 - G.u0) + 1e-12, ev = 1e-9 * (G.v1 - G.v0) + 1e-12;
        const bool per_u = G.S.k != Surf::PLANE, per_v = G.S.k == Surf::TORUS;

        if (sing) {   // (a pole: any u -- the patch's middle)
            u = 0.5 * (G.u0 + G.u1);
        }

        if (!(std::fabs(u) < 1e6 && std::fabs(v) < 1e12)) {
            return false;
        }

        auto wrap = [](double & x, double lo, double e) {  // (into [lo - e, lo - e + 2 pi))
            x = lo - e + std::fmod(std::fmod(x - (lo - e), 2 * kPi) + 2 * kPi, 2 * kPi);
        };

        if (per_u) {
            wrap(u, G.u0, eu);
        }

        if (per_v) {
            wrap(v, G.v0, ev);
        }

        if (u < G.u0 - eu || u > G.u1 + eu || v < G.v0 - ev || v > G.v1 + ev) {
            return false;
        }

        u = std::min(G.u1, std::max(G.u0, u));
        v = std::min(G.v1, std::max(G.v0, v));
        return norm(G.S.eval(u, v) - p) <= 1e3 * near;
    }

    // ---- the crossings of patch F (primitive a) with primitive b: marching squares ----
    void march(int fi, int b) {
        const Patch& F = Pt[static_cast<size_t>(fi)];
        const double Lu = norm(F.S.eval(F.u1, 0.5 * (F.v0 + F.v1)) - F.S.eval(F.u0, 0.5 * (F.v0 + F.v1))) + 1e-12;
        const double Lv = norm(F.S.eval(0.5 * (F.u0 + F.u1), F.v1) - F.S.eval(0.5 * (F.u0 + F.u1), F.v0)) + 1e-12;
        // (a half circle's chord: pi / 2 of its arc)
        const double arc_u = F.S.k == Surf::PLANE ? Lu : Lu * 1.6, arc_v = F.S.k == Surf::PLANE || F.S.k == Surf::CYL ||
                             F.S.k == Surf::CONE || F.S.k == Surf::DISK ? Lv : Lv * 1.6;
        const double cell = std::max(1e-6 * diag, hmax > 0 ? std::min(0.5 * hmax, diag / 96) : diag / 96);
        const int nu = std::max(4, std::min(800, static_cast<int>(std::ceil(arc_u / cell))));
        const int nv = std::max(4, std::min(800, static_cast<int>(std::ceil(arc_v / cell))));
        auto uvp = [&](double u, double v) {
            return F.S.eval(u, v);
        };
        auto g = [&](double u, double v) {
            return fprim(b, uvp(u, v));
        };
        std::vector<double> G(static_cast<size_t>((nu + 1) * (nv + 1)));
        size_t zeros = 0;

        for (int j = 0; j <= nv; ++j)
            for (int i = 0; i <= nu; ++i) {
                double x = g(F.u0 + (F.u1 - F.u0) * i / nu, F.v0 + (F.v1 - F.v0) * j / nv);

                if (std::fabs(x) <= 1e-6 * diag) {
                    ++zeros;
                    x = 1e-6 * diag;   // (a zero: counted positive)
                }

                G[static_cast<size_t>(j * (nu + 1) + i)] = x;
            }

        if (zeros > static_cast<size_t>((nu + 1) * (nv + 1)) / 5) {
            throw std::runtime_error("exact surface: two curved surfaces lie on each other (not supported yet: use --mode mesh)");
        }

        auto gv = [&](int i, int j) {
            return G[static_cast<size_t>(j * (nu + 1) + i)];
        };
        // the crossings on the cell edges: key 2 (j (nu + 1) + i) + (0: along u to i + 1, 1: along v to j + 1)
        std::unordered_map<int64_t, int32_t> xnode;
        auto crossing = [&](int i, int j, int dir) -> int32_t {
            const int64_t key = 2 * (static_cast<int64_t>(j) * (nu + 1) + i) + dir;
            const auto it = xnode.find(key);

            if (it != xnode.end()) {
                return it->second;
            }

            // bisection along the cell edge in (u, v)
            double ua = F.u0 + (F.u1 - F.u0) * i / nu, va = F.v0 + (F.v1 - F.v0) * j / nv;
            double ub = dir == 0 ? F.u0 + (F.u1 - F.u0) * (i + 1) / nu : ua, vb = dir == 1 ? F.v0 + (F.v1 - F.v0) * (j + 1) / nv : va;
            const double ga = gv(i, j);
            double lo = 0, hi = 1;
            const bool neg = ga < 0;

            for (int it2 = 0; it2 < 60; ++it2) {
                const double m = 0.5 * (lo + hi);
                const double x = g(ua + m * (ub - ua), va + m * (vb - va));
                ((x < 0) == neg ? lo : hi) = m;
            }

            const double m = 0.5 * (lo + hi), u = ua + m * (ub - ua), v = va + m * (vb - va);
            const A3 p = uvp(u, v);
            int32_t id;
            // on the patch's boundary (a cell edge along it, or a root at its end: the
            // crossing running along a side -- a torus cut at its equator): a node of its edge
            const double tu = 1e-9 * (F.u1 - F.u0), tv = 1e-9 * (F.v1 - F.v0);
            int s = -1;
            double sg2 = 0;

            if ((dir == 0 && (j == 0 || j == nv)) || std::fabs(v - F.v0) <= tv || std::fabs(v - F.v1) <= tv) {
                s = std::fabs(v - F.v0) < std::fabs(v - F.v1) ? 0 : 2;
                sg2 = (u - F.u0) / (F.u1 - F.u0);
            } else if ((dir == 1 && (i == 0 || i == nu)) || std::fabs(u - F.u0) <= tu || std::fabs(u - F.u1) <= tu) {
                s = std::fabs(u - F.u0) < std::fabs(u - F.u1) ? 3 : 1;
                sg2 = (v - F.v0) / (F.v1 - F.v0);
            }

            if (s >= 0 && !E[static_cast<size_t>(F.e[s])].degen) {
                const double t = std::min(1.0, std::max(0.0, F.rev[s] ? 1 - sg2 : sg2));
                id = on_edge(F.e[s], t, E[static_cast<size_t>(F.e[s])].at(t));
            } else if (s >= 0) {
                id = E[static_cast<size_t>(F.e[s])].n[0];
            } else {
                id = add(p);
            }

            xnode[key] = id;
            return id;
        };
        // the segments, then chained
        std::multimap<int32_t, int32_t> adj;
        std::set<int32_t> boundary;

        for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i) {
                const double g00 = gv(i, j), g10 = gv(i + 1, j), g11 = gv(i + 1, j + 1), g01 = gv(i, j + 1);
                int32_t x[4];
                int nx = 0;

                if ((g00 < 0) != (g10 < 0)) {
                    x[nx++] = crossing(i, j, 0);   // bottom
                }

                if ((g10 < 0) != (g11 < 0)) {
                    x[nx++] = crossing(i + 1, j, 1);   // right
                }

                if ((g01 < 0) != (g11 < 0)) {
                    x[nx++] = crossing(i, j + 1, 0);   // top
                }

                if ((g00 < 0) != (g01 < 0)) {
                    x[nx++] = crossing(i, j, 1);   // left
                }

                if (nx == 2) {
                    if (x[0] != x[1]) {   // (both at a pole: none)
                        adj.emplace(x[0], x[1]);
                        adj.emplace(x[1], x[0]);
                    }
                } else if (nx == 4) {   // a saddle: by the centre's sign
                    const double gc = g(F.u0 + (F.u1 - F.u0) * (i + 0.5) / nu, F.v0 + (F.v1 - F.v0) * (j + 0.5) / nv);

                    if ((gc < 0) == (g00 < 0)) {   // corner 00's region joins the centre: cut 10 and 01 off
                        adj.emplace(x[0], x[1]);
                        adj.emplace(x[1], x[0]);
                        adj.emplace(x[2], x[3]);
                        adj.emplace(x[3], x[2]);
                    } else {
                        adj.emplace(x[0], x[3]);
                        adj.emplace(x[3], x[0]);
                        adj.emplace(x[1], x[2]);
                        adj.emplace(x[2], x[1]);
                    }
                }
            }

        for (const auto& kv : xnode) {
            const int64_t key = kv.first;
            const int dir = static_cast<int>(key & 1);
            const int64_t ij = key >> 1;
            const int i = static_cast<int>(ij % (nu + 1)), j = static_cast<int>(ij / (nu + 1));

            if ((dir == 0 && (j == 0 || j == nv)) || (dir == 1 && (i == 0 || i == nu))) {
                boundary.insert(kv.second);
            }
        }

        std::set<std::pair<int32_t, int32_t>> used;
        auto walk = [&](int32_t s0) {
            Chain ch;
            ch.a = F.prim;
            ch.b = b;
            ch.n.push_back(s0);
            int32_t cur = s0, prev = -1;

            for (;;) {
                int32_t nxt = -1;
                const auto rg = adj.equal_range(cur);

                for (auto it = rg.first; it != rg.second; ++it)
                    if (it->second != prev && !used.count({ std::min(cur, it->second), std::max(cur, it->second) })) {
                        nxt = it->second;
                        break;
                    }

                if (nxt < 0) {
                    break;
                }

                used.insert({ std::min(cur, nxt), std::max(cur, nxt) });
                ch.n.push_back(nxt);
                prev = cur;
                cur = nxt;

                if (cur == s0) {
                    break;   // (closed)
                }
            }

            if (ch.n.size() >= 2) {
                C.push_back(ch);
            }
        };

        for (const int32_t s0 : boundary) {   // the open curves first, from the boundary
            const auto rg = adj.equal_range(s0);

            for (auto it = rg.first; it != rg.second; ++it)
                if (!used.count({ std::min(s0, it->second), std::max(s0, it->second) })) {
                    walk(s0);
                }
        }

        for (const auto& kv : adj)   // then the closed ones
            if (!used.count({ std::min(kv.first, kv.second), std::max(kv.first, kv.second) })) {
                walk(kv.first);
            }
    }

    // ---- the chains into the other primitive's patches: split at their edges -----------
    std::vector<int> patches_of(int prim, const A3& p) const {
        std::vector<int> r;

        for (size_t gi = 0; gi < Pt.size(); ++gi) {
            const Patch& G = Pt[gi];

            if (G.prim != prim) {
                continue;
            }

            double u, v;

            if (locate(G, p, u, v)) {
                r.push_back(static_cast<int>(gi));
            }
        }

        return r;
    }
    void import(Chain& ch) {
        for (size_t k = 0; k + 1 < ch.n.size(); ++k) {
            const A3 pa = at(ch.n[k]), pb = at(ch.n[k + 1]);
            const std::vector<int> ma = patches_of(ch.b, pa), mb = patches_of(ch.b, pb);
            bool common = false;

            for (const int x : ma)
                for (const int y : mb) {
                    common = common || x == y;
                }

            if (common || ma.empty() || mb.empty()) {
                continue;
            }

            // the edge of ma's patch between them: its point where primitive a's surface crosses it
            const Patch& G = Pt[static_cast<size_t>(ma[0])];
            double best = 1e300, bt = 0;
            int bs = -1;

            for (int s = 0; s < 4; ++s) {
                const Edge& e = E[static_cast<size_t>(G.e[s])];

                if (e.degen) {
                    continue;
                }

                // the root of a's field along the edge nearest the segment
                const int ns = 400;
                double fa = fprim(ch.a, e.at(0));

                for (int i = 1; i <= ns; ++i) {
                    const double fb = fprim(ch.a, e.at(static_cast<double>(i) / ns));

                    if ((fa < 0) != (fb < 0)) {
                        double lo = static_cast<double>(i - 1) / ns, hi = static_cast<double>(i) / ns;
                        const bool neg = fa < 0;

                        for (int it = 0; it < 60; ++it) {
                            const double m = 0.5 * (lo + hi);
                            ((fprim(ch.a, e.at(m)) < 0) == neg ? lo : hi) = m;
                        }

                        const A3 q = e.at(0.5 * (lo + hi));
                        const double d = norm(q - pa) + norm(q - pb);

                        if (d < best) {
                            best = d;
                            bt = 0.5 * (lo + hi);
                            bs = s;
                        }
                    }

                    fa = fb;
                }
            }

            if (bs < 0 || best > 3 * norm(pb - pa) + 10 * near) {
                continue;   // (not found: left -- the patch's triangulation copes, or reports)
            }

            const int32_t id = on_edge(G.e[bs], bt, E[static_cast<size_t>(G.e[bs])].at(bt));
            ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(k) + 1, id);
            ++k;
        }
    }

    // ---- tessellation units: a curved patch alone, or all the planar patches on one
    // plane (coincident faces -- a box's on the domain's, a cap on a face -- triangulated
    // once, together)
    struct Unit {
        std::vector<int> mem;
        bool planar = false;
        A3 o{ { 0, 0, 0 } }, ex{ { 1, 0, 0 } }, ey{ { 0, 1, 0 } }, nz{ { 0, 0, 1 } };
        std::set<std::pair<int32_t, int32_t>> drop;   // chain segments traced twice: the copies
    };
    std::vector<Unit> U;
    std::vector<int> unit_of;

    static bool planar(const Patch& G) {
        return G.S.k == Surf::PLANE || G.S.k == Surf::DISK;
    }
    void build_units() {
        unit_of.assign(Pt.size(), -1);

        for (size_t gi = 0; gi < Pt.size(); ++gi) {
            if (unit_of[gi] >= 0) {
                continue;
            }

            Unit u;
            u.mem.push_back(static_cast<int>(gi));
            unit_of[gi] = static_cast<int>(U.size());

            if (planar(Pt[gi])) {
                u.planar = true;
                u.o = Pt[gi].S.f.o;
                u.ex = Pt[gi].S.f.x;
                u.ey = Pt[gi].S.f.y;
                u.nz = unit(cross(u.ex, u.ey));

                for (size_t gj = gi + 1; gj < Pt.size(); ++gj)
                    if (unit_of[gj] < 0 && planar(Pt[gj])) {
                        const A3 n2 = unit(cross(Pt[gj].S.f.x, Pt[gj].S.f.y));

                        if (norm(cross(u.nz, n2)) < 1e-9 && std::fabs(dot(u.nz, Pt[gj].S.f.o - u.o)) <= near) {
                            u.mem.push_back(static_cast<int>(gj));
                            unit_of[gj] = static_cast<int>(U.size());
                        }
                    }
            }

            U.push_back(u);
        }
    }
    // does patch fi lie on a plane with one of primitive b's (their crossing: that one's sides)?
    bool shares_plane(int fi, int b) const {
        for (const int m : U[static_cast<size_t>(unit_of[static_cast<size_t>(fi)])].mem)
            if (Pt[static_cast<size_t>(m)].prim == b) {
                return true;
            }

        return false;
    }
    static void plane_xy(const Unit& u, const A3& p, double& x, double& y) {
        x = dot(p - u.o, u.ex);
        y = dot(p - u.o, u.ey);
    }
    size_t prims_in(const Unit& u) const {
        std::set<int> s;

        for (const int m : u.mem) {
            s.insert(Pt[static_cast<size_t>(m)].prim);
        }

        return s.size();
    }

    // ---- an edge's curve: a line, or a circle (the sides a plane can hold) ---------------
    struct Geo {
        bool line = true;
        A3 a{ { 0, 0, 0 } }, b{ { 0, 0, 0 } }, c{ { 0, 0, 0 } };
        double r = 0;
    };
    Geo geo(const Edge& e) const {
        Geo g;
        g.a = e.at(0);
        g.b = e.at(1);
        const A3 m = e.at(0.5), u = g.b - g.a, v = m - g.a, w = cross(u, v);
        const double L = norm(u);
        g.line = !(L > 0) || norm(w) / L <= near;

        if (!g.line) {   // (the circle through its ends and middle)
            g.c = g.a + (0.5 / dot(w, w)) * (dot(u, u) * cross(v, w) + dot(v, v) * cross(w, u));
            g.r = norm(g.a - g.c);
        }

        return g;
    }
    // (> 0 on one side; 0 on the curve), for a point on the plane of normal nz
    static double phi(const Geo& g, const A3& nz, const A3& p) {
        return g.line ? dot(p - g.a, unit(cross(nz, g.b - g.a))) : g.r - norm(p - g.c);
    }
    // point p's parameter along edge e (-1: not on it)
    double edge_t(const Edge& e, const A3& p) const {
        const Geo g = geo(e);
        double t;

        if (g.line) {
            const A3 d = g.b - g.a;
            t = dot(p - g.a, d) / dot(d, d);
        } else {
            double u, v;
            bool sing;
            e.S.inv(p, u, v, sing, 0, 0, false);
            const double mid = 0.5 * (e.ua + e.ub);
            u = mid + std::remainder(u - mid, 2 * kPi);
            const double du = e.ub - e.ua, dv = e.vb - e.va;
            t = ((u - e.ua) * du + (v - e.va) * dv) / (du * du + dv * dv);
        }

        return t > 0 && t < 1 && norm(e.at(t) - p) <= 10 * near ? t : -1;
    }
    // node id (at p) on edge ei too, if it lies on it (false: already there, or off it)
    bool share(int ei, const A3& p, int32_t id) {
        Edge& e = E[static_cast<size_t>(ei)];

        if (e.degen) {
            return false;
        }

        for (const int32_t n : e.n)
            if (n == id || norm(at(n) - p) <= near) {
                return false;
            }

        const double t = edge_t(e, p);

        if (t < 0) {
            return false;
        }

        const size_t k = static_cast<size_t>(std::upper_bound(e.t.begin(), e.t.end(), t) - e.t.begin());
        e.t.insert(e.t.begin() + static_cast<std::ptrdiff_t>(k), t);
        e.n.insert(e.n.begin() + static_cast<std::ptrdiff_t>(k), id);
        return true;
    }
    // do segments ab, cd (2-D) cross properly?
    static bool cross2(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy) {
        auto o2 = [](double px, double py, double qx, double qy, double rx, double ry) {
            const double d = (qx - px) * (ry - py) - (qy - py) * (rx - px);
            return d > 0 ? 1 : d < 0 ? -1 : 0;
        };
        return o2(ax, ay, bx, by, cx, cy) * o2(ax, ay, bx, by, dx, dy) < 0 &&
               o2(cx, cy, dx, dy, ax, ay) * o2(cx, cy, dx, dy, bx, by) < 0;
    }
    // t in [t0, t1] along e where f changes sign (false: it does not)
    template <class Fn>
    static bool root_on(const Edge& e, double t0, double t1, Fn f, double& t) {
        const double f0 = f(e.at(t0)), f1 = f(e.at(t1));

        if ((f0 < 0) == (f1 < 0)) {
            return false;
        }

        const bool neg = f0 < 0;

        for (int it = 0; it < 64; ++it) {
            const double m = 0.5 * (t0 + t1);
            ((f(e.at(m)) < 0) == neg ? t0 : t1) = m;
        }

        t = 0.5 * (t0 + t1);
        return true;
    }
    std::vector<int> unit_sides(const Unit& u) const {
        std::vector<int> s;

        for (const int m : u.mem)
            for (int k = 0; k < 4; ++k) {
                const int ei = Pt[static_cast<size_t>(m)].e[k];

                if (!E[static_cast<size_t>(ei)].degen && std::find(s.begin(), s.end(), ei) == s.end()) {
                    s.push_back(ei);
                }
            }

        return s;
    }
    // on a plane of patches of several primitives: their sides split where they cross
    // or touch each other, the same nodes on both
    void junctions() {
        for (Unit& u : U) {
            if (!u.planar || prims_in(u) < 2) {
                continue;
            }

            const std::vector<int> S = unit_sides(u);

            for (size_t i = 0; i < S.size(); ++i)
                for (size_t j = 0; j < S.size(); ++j) {
                    if (i == j) {
                        continue;
                    }

                    const Geo g2 = geo(E[static_cast<size_t>(S[j])]);
                    bool again = true;

                    for (int pass = 0; again && pass < 64; ++pass) {
                        again = false;
                        const Edge& e1 = E[static_cast<size_t>(S[i])];
                        const Edge& e2 = E[static_cast<size_t>(S[j])];

                        for (size_t k = 0; k + 1 < e1.n.size() && !again; ++k)
                            for (size_t l = 0; l + 1 < e2.n.size() && !again; ++l) {
                                double ax, ay, bx, by, cx, cy, dx, dy;
                                plane_xy(u, at(e1.n[k]), ax, ay);
                                plane_xy(u, at(e1.n[k + 1]), bx, by);
                                plane_xy(u, at(e2.n[l]), cx, cy);
                                plane_xy(u, at(e2.n[l + 1]), dx, dy);

                                if (!cross2(ax, ay, bx, by, cx, cy, dx, dy)) {
                                    continue;
                                }

                                double t;
                                const A3 nz = u.nz;

                                if (!root_on(e1, e1.t[k], e1.t[k + 1], [&](const A3 & p) {
                                return phi(g2, nz, p);
                                }, t)) {
                                    continue;
                                }

                                const A3 p = e1.at(t);
                                const int32_t id = on_edge(S[i], t, p);
                                share(S[j], p, id);
                                again = true;
                            }
                    }
                }

            // (a node of one on another: a corner on a side, collinear sides overlapping)
            for (int pass = 0; pass < 2; ++pass)
                for (size_t i = 0; i < S.size(); ++i)
                    for (size_t j = 0; j < S.size(); ++j) {
                        if (i == j) {
                            continue;
                        }

                        const Geo g1 = geo(E[static_cast<size_t>(S[i])]);
                        const std::vector<int32_t> ns = E[static_cast<size_t>(S[j])].n;

                        for (const int32_t n : ns)
                            if (std::fabs(phi(g1, u.nz, at(n))) <= 10 * near) {
                                share(S[i], at(n), n);
                            }
                    }
        }
    }

    // ---- the chains' segments a unit holds: (chain, index, member) ----------------------
    struct Seg {
        int c;
        size_t k;
        int m;
    };
    int other(const Chain& ch, int m) const {
        return ch.a == Pt[static_cast<size_t>(m)].prim ? ch.b : ch.a;
    }
    std::vector<Seg> unit_segs(const Unit& u) const {
        std::vector<Seg> r;

        for (size_t ci = 0; ci < C.size(); ++ci) {
            const Chain& ch = C[ci];
            std::vector<int> own(ch.n.size() > 0 ? ch.n.size() - 1 : 0, -1);

            for (const int m : u.mem) {
                const Patch& G = Pt[static_cast<size_t>(m)];

                if (ch.a != G.prim && ch.b != G.prim) {
                    continue;
                }

                std::vector<char> on(ch.n.size());

                for (size_t k = 0; k < ch.n.size(); ++k) {
                    double uu, vv;
                    on[k] = locate(G, at(ch.n[k]), uu, vv);
                }

                for (size_t k = 0; k + 1 < ch.n.size(); ++k)
                    if (own[k] < 0 && on[k] && on[k + 1] && ch.n[k] != ch.n[k + 1]) {
                        own[k] = m;
                    }
            }

            for (size_t k = 0; k < own.size(); ++k)
                if (own[k] >= 0 && !u.drop.count({ std::min(ch.n[k], ch.n[k + 1]), std::max(ch.n[k], ch.n[k + 1]) })) {
                    r.push_back({ static_cast<int>(ci), k, own[k] });
                }
        }

        return r;
    }
    // a unit's 2-D coordinates of point p (a plane's; a patch's (u, v))
    void xy(const Unit& u, const A3& p, double& x, double& y) const {
        if (u.planar) {
            plane_xy(u, p, x, y);
        } else {
            locate(Pt[static_cast<size_t>(u.mem[0])], p, x, y);
        }
    }
    // on a plane of several primitives' patches: a chain split where it crosses the
    // others' sides (the crossing: the root of its other primitive's field along the side)
    void side_splits() {
        for (Unit& u : U) {
            if (!u.planar || prims_in(u) < 2) {
                continue;
            }

            const std::vector<int> S = unit_sides(u);

            for (int pass = 0; pass < 1000; ++pass) {
                bool found = false;
                const std::vector<Seg> R = unit_segs(u);

                for (size_t s = 0; s < R.size() && !found; ++s) {
                    Chain& ch = C[static_cast<size_t>(R[s].c)];
                    const Patch& G = Pt[static_cast<size_t>(R[s].m)];
                    const int q = other(ch, R[s].m);
                    const int32_t na = ch.n[R[s].k], nb = ch.n[R[s].k + 1];
                    double ax, ay, bx, by;
                    plane_xy(u, at(na), ax, ay);
                    plane_xy(u, at(nb), bx, by);

                    for (const int ei : S) {
                        if (ei == G.e[0] || ei == G.e[1] || ei == G.e[2] || ei == G.e[3]) {
                            continue;
                        }

                        const Edge& e = E[static_cast<size_t>(ei)];

                        for (size_t l = 0; l + 1 < e.n.size() && !found; ++l) {
                            double cx, cy, dx, dy;
                            plane_xy(u, at(e.n[l]), cx, cy);
                            plane_xy(u, at(e.n[l + 1]), dx, dy);
                            double t;

                            if (!cross2(ax, ay, bx, by, cx, cy, dx, dy) ||
                            !root_on(e, e.t[l], e.t[l + 1], [&](const A3 & p) {
                            return fprim(q, p);
                            }, t)) {
                                continue;
                            }

                            const int32_t id = on_edge(ei, t, e.at(t));

                            if (id != na && id != nb) {
                                ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(R[s].k) + 1, id);
                                found = true;
                            }
                        }

                        if (found) {
                            break;
                        }
                    }
                }

                if (!found) {
                    break;
                }
            }
        }
    }
    // a chain running along a patch's edge (a torus cut at its equator, a sphere at a
    // seam): the chain's nodes on the edge, the edge's on the chain -- one polyline
    void merge_edges() {
        for (Chain& ch : C) {
            std::set<int> done;
            std::vector<int> along;

            for (const Patch& G : Pt) {
                if (G.prim != ch.a && G.prim != ch.b) {
                    continue;
                }

                for (const int ei : G.e) {
                    if (E[static_cast<size_t>(ei)].degen || !done.insert(ei).second) {
                        continue;
                    }

                    size_t on = 0;

                    for (int32_t& n : ch.n) {
                        const A3 p = at(n);
                        const std::vector<int32_t>& en = E[static_cast<size_t>(ei)].n;
                        int32_t same = -1;   // (an edge node there already -- its corner: that one)

                        for (const int32_t m : en)
                            if (m != n && norm(at(m) - p) <= near) {
                                same = m;
                            }

                        if (same >= 0) {
                            n = same;
                            ++on;
                        } else if (std::find(en.begin(), en.end(), n) != en.end() || edge_t(E[static_cast<size_t>(ei)], p) >= 0) {
                            share(ei, p, n);
                            ++on;
                        }
                    }

                    if (on < 2) {
                        continue;   // (crossing it, not along it)
                    }

                    along.push_back(ei);

                    // (the edge's nodes between two of the chain's on it, by the edge's parameter)
                    const Edge& e = E[static_cast<size_t>(ei)];
                    auto tof = [&](int32_t n) {
                        const auto it = std::find(e.n.begin(), e.n.end(), n);
                        return it == e.n.end() ? -1.0 : e.t[static_cast<size_t>(it - e.n.begin())];
                    };

                    for (size_t k = 0; k + 1 < ch.n.size(); ++k) {
                        const double ta = tof(ch.n[k]), tb = tof(ch.n[k + 1]);

                        if (ta < 0 || tb < 0) {
                            continue;
                        }

                        std::vector<int32_t> mid;

                        for (size_t j = 0; j < e.n.size(); ++j)
                            if (e.t[j] > std::min(ta, tb) && e.t[j] < std::max(ta, tb)) {
                                mid.push_back(e.n[j]);
                            }

                        if (ta > tb) {
                            std::reverse(mid.begin(), mid.end());
                        }

                        ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(k) + 1, mid.begin(), mid.end());
                        k += mid.size();
                    }
                }
            }

            // (a segment from one edge to the next, over the corner they share: the corner and
            // both edges' nodes between)
            auto pos = [&](int ei, int32_t n) {
                const Edge& e = E[static_cast<size_t>(ei)];
                const auto it = std::find(e.n.begin(), e.n.end(), n);
                return it == e.n.end() ? -1 : static_cast<int>(it - e.n.begin());
            };

            for (size_t k = 0; k + 1 < ch.n.size(); ++k) {
                const int32_t a = ch.n[k], b = ch.n[k + 1];
                std::vector<int32_t> mid;

                for (const int e1 : along)
                    for (const int e2 : along) {
                        const int ia = pos(e1, a), ib = pos(e2, b);

                        if (e1 == e2 || ia < 0 || ib < 0 || pos(e1, b) >= 0 || pos(e2, a) >= 0 || !mid.empty()) {
                            continue;
                        }

                        const std::vector<int32_t>& n1 = E[static_cast<size_t>(e1)].n;
                        const std::vector<int32_t>& n2 = E[static_cast<size_t>(e2)].n;

                        for (const int32_t c : {
                                    n1.front(), n1.back()
                                }) {
                            const int jc1 = pos(e1, c), jc2 = pos(e2, c);

                            if (jc2 < 0 || !mid.empty()) {
                                continue;
                            }

                            for (int j = ia; j != jc1;) {   // a -> c along e1
                                j += jc1 > ia ? 1 : -1;
                                mid.push_back(n1[static_cast<size_t>(j)]);
                            }

                            for (int j = jc2; j != ib;) {   // c -> b along e2
                                j += ib > jc2 ? 1 : -1;

                                if (j != ib) {
                                    mid.push_back(n2[static_cast<size_t>(j)]);
                                }
                            }
                        }
                    }

                ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(k) + 1, mid.begin(), mid.end());
                k += mid.size();
            }
        }
    }
    // point p's distance to segment ab
    static double seg_dist(const A3& p, const A3& a, const A3& b) {
        const A3 d = b - a;
        const double l2 = dot(d, d), t = l2 > 0 ? std::max(0.0, std::min(1.0, dot(p - a, d) / l2)) : 0.0;
        return norm(p - (a + t * d));
    }
    // two chains tracing the same curve in a unit (a coincident face's crossing with a
    // third surface, found from both): the later's copy dropped, its nodes put in the
    // earlier (so both ends meet)
    void dedup() {
        for (Unit& u : U) {
            if (!(u.planar ? prims_in(u) >= 2 : true)) {
                continue;
            }

            std::vector<Seg> R = unit_segs(u);
            std::map<int, std::vector<Seg>> by;

            for (const Seg& s : R) {
                by[s.c].push_back(s);
            }

            if (by.size() < 2) {
                continue;
            }

            for (auto i2 = by.begin(); i2 != by.end(); ++i2)
                for (auto i1 = by.begin(); i1 != i2; ++i1) {
                    Chain& A = C[static_cast<size_t>(i1->first)];
                    const Chain& B = C[static_cast<size_t>(i2->first)];

                    if (other(A, i1->second[0].m) != other(B, i2->second[0].m)) {
                        continue;   // (different curves)
                    }

                    std::vector<int32_t> moved;

                    for (const Seg& s : i2->second) {
                        const int32_t a = B.n[s.k], b = B.n[s.k + 1];
                        const A3 pa = at(a), pb = at(b), pm = 0.5 * (pa + pb);
                        const double d = 0.1 * norm(pb - pa) + 10 * near;
                        double da = 1e300, db = 1e300, dm = 1e300;

                        for (size_t k = 0; k + 1 < A.n.size(); ++k) {
                            const A3 qa = at(A.n[k]), qb = at(A.n[k + 1]);
                            da = std::min(da, seg_dist(pa, qa, qb));
                            db = std::min(db, seg_dist(pb, qa, qb));
                            dm = std::min(dm, seg_dist(pm, qa, qb));
                        }

                        if (da < d && db < d && dm < d) {
                            u.drop.insert({ std::min(a, b), std::max(a, b) });
                            moved.push_back(a);
                            moved.push_back(b);
                        }
                    }

                    for (const int32_t n : moved) {   // (into A, where it passes)
                        if (std::find(A.n.begin(), A.n.end(), n) != A.n.end()) {
                            continue;
                        }

                        const A3 p = at(n);
                        double best = 1e300;
                        size_t bk = 0;
                        bool dup = false;

                        for (size_t k = 0; k + 1 < A.n.size(); ++k) {
                            dup = dup || norm(at(A.n[k]) - p) <= near;
                            const double dd = seg_dist(p, at(A.n[k]), at(A.n[k + 1]));

                            if (dd < best) {
                                best = dd;
                                bk = k;
                            }
                        }

                        if (!dup && best < 0.1 * norm(at(A.n[bk + 1]) - at(A.n[bk])) + 10 * near) {
                            A.n.insert(A.n.begin() + static_cast<std::ptrdiff_t>(bk) + 1, n);
                        }
                    }
                }
        }
    }

    // ---- triple points: two chains crossing in a unit -----------------------------------
    // (u, v) on surface S where two primitives' surfaces both cross it: Newton from (u, v)
    bool triple(const Surf& S, int qa, int qb, double& u, double& v) const {
        const double eu = 1e-7 * (1 + std::fabs(u)), ev = 1e-7 * (1 + std::fabs(v));

        for (int it = 0; it < 40; ++it) {
            const double f1 = fprim(qa, S.eval(u, v)), f2 = fprim(qb, S.eval(u, v));

            if (std::fabs(f1) < 1e-9 * diag && std::fabs(f2) < 1e-9 * diag) {
                return true;
            }

            const double a11 = (fprim(qa, S.eval(u + eu, v)) - fprim(qa, S.eval(u - eu, v))) / (2 * eu);
            const double a12 = (fprim(qa, S.eval(u, v + ev)) - fprim(qa, S.eval(u, v - ev))) / (2 * ev);
            const double a21 = (fprim(qb, S.eval(u + eu, v)) - fprim(qb, S.eval(u - eu, v))) / (2 * eu);
            const double a22 = (fprim(qb, S.eval(u, v + ev)) - fprim(qb, S.eval(u, v - ev))) / (2 * ev);
            const double det = a11 * a22 - a12 * a21;

            if (!(std::fabs(det) > 1e-30)) {
                return false;
            }

            u -= (a22 * f1 - a12 * f2) / det;
            v -= (a11 * f2 - a21 * f1) / det;
        }

        return std::fabs(fprim(qa, S.eval(u, v))) < 1e-7 * diag && std::fabs(fprim(qb, S.eval(u, v))) < 1e-7 * diag;
    }
    std::vector<int32_t> tp;   // the triple points
    // a triple point on every chain whose two surfaces pass through it (found where two
    // of its three curves cross; the third only touches it there)
    void thread_triples() {
        for (Chain& ch : C)
            for (const int32_t t : tp) {
                if (std::find(ch.n.begin(), ch.n.end(), t) != ch.n.end()) {
                    continue;
                }

                const A3 p = at(t);

                if (std::fabs(fprim(ch.a, p)) > 1e-6 * diag || std::fabs(fprim(ch.b, p)) > 1e-6 * diag) {
                    continue;
                }

                double best = 1e300;
                size_t bk = 0;

                for (size_t k = 0; k + 1 < ch.n.size(); ++k) {
                    const double d = seg_dist(p, at(ch.n[k]), at(ch.n[k + 1]));

                    if (d < best) {
                        best = d;
                        bk = k;
                    }
                }

                if (ch.n.size() >= 2 && best < 0.2 * norm(at(ch.n[bk + 1]) - at(ch.n[bk])) + 10 * near) {
                    ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(bk) + 1, t);
                }
            }
    }
    // the chains thinned to the edges' sampling (the chord tolerance, the turn, --size):
    // traced at the marching cells, they are denser than the faces round them need; a
    // node an edge or another chain holds stays
    void thin_chains() {
        std::unordered_map<int32_t, int> uses;

        for (const Edge& e : E)
            for (const int32_t n : e.n) {
                uses[n] += 2;
            }

        for (const Chain& ch : C) {
            std::set<int32_t> own(ch.n.begin(), ch.n.end());

            for (const int32_t n : own) {
                ++uses[n];
            }
        }

        for (const int32_t t : tp) {
            uses[t] += 2;
        }

        const double cmin = std::cos(amax);

        for (Chain& ch : C) {
            if (ch.n.size() < 3) {
                continue;
            }

            std::vector<int32_t> out{ ch.n[0] };
            size_t a = 0;

            while (a + 1 < ch.n.size()) {
                size_t j = a + 1;   // (the farthest node the chord a -> j can reach)

                while (j + 1 < ch.n.size() && uses[ch.n[j]] < 2) {
                    const A3 pa = at(ch.n[a]), pb = at(ch.n[j + 1]);
                    const double L = norm(pb - pa);
                    bool fine = !(hmax > 0) || L <= hmax;

                    for (size_t k = a + 1; fine && k <= j; ++k) {
                        fine = seg_dist(at(ch.n[k]), pa, pb) <= tol;
                    }

                    // (the turn: the chord against the first and last pieces it replaces)
                    const A3 d0 = at(ch.n[a + 1]) - pa, d1 = pb - at(ch.n[j]);

                    if (fine && L > 0 && norm(d0) > 0 && norm(d1) > 0) {
                        fine = dot(d0, pb - pa) / (norm(d0) * L) >= cmin && dot(d1, pb - pa) / (norm(d1) * L) >= cmin;
                    }

                    if (!fine) {
                        break;
                    }

                    ++j;
                }

                out.push_back(ch.n[j]);
                a = j;
            }

            ch.n.swap(out);
        }
    }
    Surf unit_surf(const Unit& u) const {
        if (!u.planar) {
            return Pt[static_cast<size_t>(u.mem[0])].S;
        }

        Surf S;
        S.k = Surf::PLANE;
        S.f.o = u.o;
        S.f.x = u.ex;
        S.f.y = u.ey;
        S.f.z = u.nz;
        return S;
    }
    void triples() {
        std::set<std::array<int32_t, 4>> passed;

        for (Unit& u : U) {
            const Surf S = unit_surf(u);

            for (int pass = 0; pass < 200; ++pass) {
                const std::vector<Seg> R = unit_segs(u);
                bool found = false;

                for (size_t x = 0; x < R.size() && !found; ++x)
                    for (size_t y = x + 1; y < R.size() && !found; ++y) {
                        if (R[x].c == R[y].c) {
                            continue;
                        }

                        const Chain& A = C[static_cast<size_t>(R[x].c)], &B = C[static_cast<size_t>(R[y].c)];
                        const int qa = other(A, R[x].m), qb = other(B, R[y].m);

                        if (qa == qb) {
                            continue;
                        }

                        const int32_t a0 = A.n[R[x].k], a1 = A.n[R[x].k + 1], b0 = B.n[R[y].k], b1 = B.n[R[y].k + 1];

                        if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 || passed.count({ { a0, a1, b0, b1 } })) {
                            continue;
                        }

                        double ua0, va0, ua1, va1, ub0, vb0, ub1, vb1;
                        xy(u, at(a0), ua0, va0);
                        xy(u, at(a1), ua1, va1);
                        xy(u, at(b0), ub0, vb0);
                        xy(u, at(b1), ub1, vb1);

                        if (!cross2(ua0, va0, ua1, va1, ub0, vb0, ub1, vb1)) {
                            continue;
                        }

                        double uu = 0.25 * (ua0 + ua1 + ub0 + ub1), vv = 0.25 * (va0 + va1 + vb0 + vb1);

                        if (!triple(S, qa, qb, uu, vv)) {
                            continue;
                        }

                        const A3 p = S.eval(uu, vv);
                        int32_t id = -1;

                        for (const Chain& ch : C)   // (found already from another unit)
                            for (const int32_t n : ch.n)
                                if (id < 0 && norm(at(n) - p) <= near) {
                                    id = n;
                                }

                        if (id < 0) {
                            id = add(p);
                            ++st.triple;
                            tp.push_back(id);
                        }

                        // (into each chain not holding it yet; a pair both already hold: passed)
                        // (at the chain's segment nearest it: the polylines may cross a segment off)
                        auto put = [&](Chain & ch, size_t k) {
                            if (std::find(ch.n.begin(), ch.n.end(), id) != ch.n.end()) {
                                return false;
                            }

                            double best = seg_dist(p, at(ch.n[k]), at(ch.n[k + 1]));

                            for (size_t j = k >= 2 ? k - 2 : 0; j + 1 < ch.n.size() && j <= k + 2; ++j) {
                                const double d = seg_dist(p, at(ch.n[j]), at(ch.n[j + 1]));

                                if (d < best) {
                                    best = d;
                                    k = j;
                                }
                            }

                            ch.n.insert(ch.n.begin() + static_cast<std::ptrdiff_t>(k) + 1, id);
                            return true;
                        };
                        const bool pa = put(C[static_cast<size_t>(R[x].c)], R[x].k);
                        const bool pb = put(C[static_cast<size_t>(R[y].c)], R[y].k);

                        if (!pa && !pb) {
                            passed.insert({ { a0, a1, b0, b1 } });
                            continue;
                        }

                        found = true;
                    }

                if (!found) {
                    break;
                }
            }
        }
    }

    // ---- a unit triangulated, its pieces kept by the labels across them -----------------
    void tessellate(const Unit& un, std::vector<int32_t>& tris, std::vector<int32_t>& tlab) {
        const Patch& G = Pt[static_cast<size_t>(un.mem[0])];
        // the metric: a plane's own; a patch's u, v scaled to mm, the largest across it (a
        // sphere's equator)
        double su = 1, sv = 1;

        if (!un.planar) {
            const double du = 1e-4 * (G.u1 - G.u0), dv = 1e-4 * (G.v1 - G.v0);
            su = sv = 0;

            for (int k = 0; k <= 8; ++k) {
                const double uk = G.u0 + (G.u1 - G.u0) * (0.5 + 0.49 * std::cos(kPi * k / 8));
                const double vk = G.v0 + (G.v1 - G.v0) * (0.5 + 0.49 * std::cos(kPi * k / 8));
                const double um = 0.5 * (G.u0 + G.u1), vm = 0.5 * (G.v0 + G.v1);
                su = std::max(su, norm(G.S.eval(um + du, vk) - G.S.eval(um - du, vk)) / (2 * du));
                sv = std::max(sv, norm(G.S.eval(uk, vm + dv) - G.S.eval(uk, vm - dv)) / (2 * dv));
            }

            if (!(su > 0)) {
                su = 1;
            }

            if (!(sv > 0)) {
                sv = 1;
            }
        }

        const Surf S = unit_surf(un);
        auto p3 = [&](double x, double y) {   // (2-D -> the surface)
            return S.eval(x / su, y / sv);
        };
        auto inside = [&](const A3 & p) {  // (on one of the unit's patches)
            for (const int m : un.mem) {
                double uu, vv;

                if (locate(Pt[static_cast<size_t>(m)], p, uu, vv)) {
                    return true;
                }
            }

            return false;
        };
        std::vector<double> P;
        std::vector<int32_t> pid;
        std::map<std::pair<long long, long long>, int> key;
        const double q = 1e-9 * diag;
        auto point = [&](double x, double y, int32_t id) {   // (x, y: scaled)
            const std::pair<long long, long long> k(std::llround(x / q), std::llround(y / q));
            const auto it = key.find(k);

            if (it != key.end()) {
                return it->second;
            }

            const int i = static_cast<int>(pid.size());
            key[k] = i;
            P.push_back(x);
            P.push_back(y);
            pid.push_back(id);
            return i;
        };
        std::unordered_map<int32_t, int> byid;   // (a node the sides placed: there)
        auto node = [&](int32_t id) {
            const auto it = byid.find(id);

            if (it != byid.end()) {
                return it->second;
            }

            double x, y;
            xy(un, at(id), x, y);
            const int i = point(x * su, y * sv, id);
            byid.emplace(id, i);
            return i;
        };
        std::vector<int> segs;
        std::set<std::pair<int, int>> seen;
        auto seg = [&](int a, int b) {
            if (a != b && seen.insert({ std::min(a, b), std::max(a, b) }).second) {
                segs.push_back(a);
                segs.push_back(b);
            }
        };

        // the sides
        if (un.planar) {
            for (const int ei : unit_sides(un)) {
                const Edge& e = E[static_cast<size_t>(ei)];

                for (size_t k = 0; k + 1 < e.n.size(); ++k) {
                    seg(node(e.n[k]), node(e.n[k + 1]));
                }
            }
        } else
            for (int s = 0; s < 4; ++s) {
                const Edge& e = E[static_cast<size_t>(G.e[s])];
                std::vector<int> ix;

                if (e.degen) {   // a pole: points along the side, all its one node
                    for (int k = 0; k <= 16; ++k) {
                        double uu, vv;
                        G.side_uv(s, k / 16.0, uu, vv);
                        ix.push_back(point(uu * su, vv * sv, e.n[0]));
                    }
                } else
                    for (size_t k = 0; k < e.n.size(); ++k) {
                        const double t = G.rev[s] ? 1 - e.t[e.n.size() - 1 - k] : e.t[k];
                        const int32_t id = G.rev[s] ? e.n[e.n.size() - 1 - k] : e.n[k];
                        double uu, vv;
                        G.side_uv(s, t, uu, vv);
                        ix.push_back(point(uu * su, vv * sv, id));
                        byid.emplace(id, ix.back());
                    }

                for (size_t k = 0; k + 1 < ix.size(); ++k) {
                    seg(ix[k], ix[k + 1]);
                }
            }

        // the chains in it
        for (const Seg& s : unit_segs(un)) {
            const Chain& ch = C[static_cast<size_t>(s.c)];
            seg(node(ch.n[s.k]), node(ch.n[s.k + 1]));
        }

        // interior points: a lattice clear of the constraints (--size, and a curved
        // surface's tolerance)
        {
            double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;

            for (size_t i = 0; i < pid.size(); ++i) {
                x0 = std::min(x0, P[2 * i]);
                x1 = std::max(x1, P[2 * i]);
                y0 = std::min(y0, P[2 * i + 1]);
                y1 = std::max(y1, P[2 * i + 1]);
            }

            const double Wu = x1 - x0, Wv = y1 - y0;
            double cellg = hmax > 0 ? hmax : std::max(Wu, Wv);

            if (!un.planar) {   // the tolerance on a curved one: a cell's two right
                // triangles, circumradius g / sqrt(2), sag g^2 / (4 R)
                const double R = std::max(1e-9, G.S.r);
                cellg = std::min(cellg, std::max(1e-6, 2 * std::sqrt(R * tol)));
                cellg = std::min(cellg, R * amax);
            }

            int nu = std::max(1, static_cast<int>(std::floor(Wu / cellg))), nv = std::max(1, static_cast<int>(std::floor(Wv / cellg)));

            while (static_cast<double>(nu) * nv > 40000.0) {
                nu = std::max(1, nu * 3 / 4);
                nv = std::max(1, nv * 3 / 4);
            }

            // (rows of a hexagonal lattice, a little jittered in the plane: no four points
            // on a circle, so the tets' Delaunay faces are these triangles)
            nv = std::max(1, static_cast<int>(std::floor(nv / 0.866)));
            const double gx = Wu / nu, gy = Wv / nv, clear = 0.5 * std::min(gx, gy);
            const uint32_t salt = static_cast<uint32_t>(un.mem[0]);

            for (int j = 1; j < nv && Wu > 0 && Wv > 0; ++j)
                for (int i = 1; i < nu + (j & 1); ++i) {
                    uint32_t hs = salt * 2654435761u ^ static_cast<uint32_t>(i) * 40503u ^ static_cast<uint32_t>(j) * 2246822519u;
                    hs ^= hs >> 13;
                    hs *= 3266489917u;
                    hs ^= hs >> 16;
                    const double jx = ((hs & 0xffff) / 65535.0 - 0.5) * 0.1, jy = ((hs >> 16) / 65535.0 - 0.5) * 0.1;
                    const double x = x0 + gx * (i - 0.5 * (j & 1) + jx), y = y0 + gy * (j + jy);
                    bool nearc = false;

                    for (size_t k = 0; k + 1 < segs.size() && !nearc; k += 2) {
                        const double ax = P[2 * static_cast<size_t>(segs[k])], ay = P[2 * static_cast<size_t>(segs[k]) + 1];
                        const double bx = P[2 * static_cast<size_t>(segs[k + 1])], by = P[2 * static_cast<size_t>(segs[k + 1]) + 1];
                        const double ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
                        const double t = l2 > 0 ? std::max(0.0, std::min(1.0, ((x - ax) * ex + (y - ay) * ey) / l2)) : 0.0;
                        const double dx = x - (ax + t * ex), dy = y - (ay + t * ey);
                        nearc = dx * dx + dy * dy < clear * clear;
                    }

                    if (!nearc && (!un.planar || inside(p3(x, y)))) {
                        point(x, y, -1);
                    }
                }
        }

        std::vector<int> T, comp;
        std::vector<char> outc;

        try {
            cdt2d(P, segs, T, comp, outc);
        } catch (const std::exception& ex) {
            if (std::getenv("V2M_CSG_DEBUG")) {   // (which segments cross)
                std::fprintf(stderr, "[csg-debug] unit of patch %d (%zu patches, %s): %s\n", un.mem[0], un.mem.size(),
                             un.planar ? "planar" : "curved", ex.what());

                for (size_t i = 0; i + 1 < segs.size(); i += 2)
                    for (size_t j = i + 2; j + 1 < segs.size(); j += 2) {
                        const size_t a = static_cast<size_t>(segs[i]), b = static_cast<size_t>(segs[i + 1]);
                        const size_t c = static_cast<size_t>(segs[j]), d = static_cast<size_t>(segs[j + 1]);

                        if (cross2(P[2 * a], P[2 * a + 1], P[2 * b], P[2 * b + 1], P[2 * c], P[2 * c + 1], P[2 * d], P[2 * d + 1])) {
                            std::fprintf(stderr, "[csg-debug]   %d-%d x %d-%d\n", pid[a], pid[b], pid[c], pid[d]);
                        }
                    }
            }

            throw;
        }

        for (size_t i = 0; i < pid.size(); ++i)
            if (pid[i] < 0) {
                pid[i] = add(p3(P[2 * i], P[2 * i + 1]));
            }

        auto centroid = [&](size_t t, double & x, double & y) {
            const int* v = &T[3 * t];
            x = (P[2 * static_cast<size_t>(v[0])] + P[2 * static_cast<size_t>(v[1])] + P[2 * static_cast<size_t>(v[2])]) / 3;
            y = (P[2 * static_cast<size_t>(v[0]) + 1] + P[2 * static_cast<size_t>(v[1]) + 1] + P[2 * static_cast<size_t>(v[2]) + 1]) / 3;
        };
        auto normal = [&](double x, double y) {
            return un.planar ? un.nz : G.S.normal(x / su, y / sv);
        };
        // each piece: the labels across it, at its largest triangle
        std::map<int, std::pair<int, int>> keep;   // component -> (behind, in front) labels; (-1, -1) dropped
        std::map<int, std::pair<double, size_t>> big;

        for (size_t t = 0; t < comp.size(); ++t) {
            const int* v = &T[3 * t];
            const double ar = std::fabs((P[2 * static_cast<size_t>(v[1])] - P[2 * static_cast<size_t>(v[0])]) *
                                        (P[2 * static_cast<size_t>(v[2]) + 1] - P[2 * static_cast<size_t>(v[0]) + 1]) -
                                        (P[2 * static_cast<size_t>(v[2])] - P[2 * static_cast<size_t>(v[0])]) *
                                        (P[2 * static_cast<size_t>(v[1]) + 1] - P[2 * static_cast<size_t>(v[0]) + 1]));
            auto& bg = big[comp[t]];

            if (ar > bg.first) {
                bg = { ar, t };
            }
        }

        for (const auto& kv : big) {
            ++st.pieces;
            double x, y;
            centroid(kv.second.second, x, y);
            const A3 p = p3(x, y);

            if (outc[static_cast<size_t>(kv.first)] || (un.planar && !inside(p))) {
                keep[kv.first] = { -1, -1 };
                continue;
            }

            const A3 n = normal(x, y);
            const double eps = 1e-4 * diag;
            const int la = label(p - eps * n), lb = label(p + eps * n);
            keep[kv.first] = la == lb ? std::make_pair(-1, -1) : std::make_pair(la, lb);
            st.kept += la != lb;
        }

        for (size_t t = 0; t < comp.size(); ++t) {
            const auto& k = keep[comp[t]];

            if (k.first < 0) {
                continue;
            }

            int32_t a = pid[static_cast<size_t>(T[3 * t])], b = pid[static_cast<size_t>(T[3 * t + 1])],
                    c = pid[static_cast<size_t>(T[3 * t + 2])];

            if (a == b || b == c || a == c) {
                continue;   // (at a pole)
            }

            // its normal from the region behind to the one in front
            double x, y;
            centroid(t, x, y);

            if (dot(cross(at(b) - at(a), at(c) - at(a)), normal(x, y)) < 0) {
                std::swap(b, c);
            }

            tris.insert(tris.end(), { a, b, c });
            tlab.push_back(k.first);
            tlab.push_back(k.second);
        }
    }

    // nodes at one place (the same point found from two sides): one
    void weld(Mesh& m) const {
        const size_t n = X.size() / 3;
        std::vector<size_t> ord(n);

        for (size_t i = 0; i < n; ++i) {
            ord[i] = i;
        }

        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
            return X[3 * a] < X[3 * b];
        });
        std::vector<int32_t> to(n);

        for (size_t i = 0; i < n; ++i) {
            to[i] = static_cast<int32_t>(i);
        }

        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n && X[3 * ord[j]] - X[3 * ord[i]] <= near; ++j)
                if (norm(at(static_cast<int32_t>(ord[i])) - at(static_cast<int32_t>(ord[j]))) <= near) {
                    int32_t a = static_cast<int32_t>(ord[i]), b = static_cast<int32_t>(ord[j]);

                    while (to[static_cast<size_t>(a)] != a) {
                        a = to[static_cast<size_t>(a)];
                    }

                    while (to[static_cast<size_t>(b)] != b) {
                        b = to[static_cast<size_t>(b)];
                    }

                    to[static_cast<size_t>(std::max(a, b))] = std::min(a, b);
                }

        std::vector<int32_t> tr;
        std::vector<int32_t> tl;
        std::set<std::array<int32_t, 3>> have;

        for (size_t t = 0; t + 2 < m.tris.size(); t += 3) {
            int32_t v[3];

            for (int k = 0; k < 3; ++k) {
                int32_t a = m.tris[t + static_cast<size_t>(k)];

                while (to[static_cast<size_t>(a)] != a) {
                    a = to[static_cast<size_t>(a)];
                }

                v[k] = a;
            }

            std::array<int32_t, 3> s{ { v[0], v[1], v[2] } };
            std::sort(s.begin(), s.end());

            if (v[0] == v[1] || v[1] == v[2] || v[0] == v[2] || !have.insert(s).second) {
                continue;
            }

            tr.insert(tr.end(), v, v + 3);
            tl.push_back(m.tri_labels[2 * (t / 3)]);
            tl.push_back(m.tri_labels[2 * (t / 3) + 1]);
        }

        m.tris.swap(tr);
        m.tri_labels.swap(tl);
    }

    Mesh run() {
        np = static_cast<int>(sc.pcode.size());
        nl = sc.nlab;

        if (np == 0) {
            throw std::runtime_error("exact surface: no primitives");
        }

        diag = norm(A3{ { sc.hi[0] - sc.lo[0], sc.hi[1] - sc.lo[1], sc.hi[2] - sc.lo[2] } });

        if (!(diag > 0)) {
            diag = 1;
        }

        tol = o.tol > 0 ? o.tol : 5e-4 * diag;
        amax = std::max(1.0, std::min(60.0, o.angle)) * kPi / 180;
        hmax = o.size;
        near = 1e-7 * diag;
        st.tol = tol;

        // the fields: each primitive's alone; the labels' exact (no blend, no gap closing,
        // world coordinates, no brick table)
        for (int q = 0; q < np; ++q) {
            const std::vector<float>& c = sc.pcode[static_cast<size_t>(q)];
            Field F;
            F.type = c.size() >= 2 ? static_cast<int>(c[1]) : 0;
            const float* x = c.size() >= 2 ? &c[2] : nullptr;

            if (F.type == V2M_SDF_BOX || F.type == V2M_SDF_CYL || F.type == V2M_SDF_CONE) {
                F.a = { { x[0], x[1], x[2] } };
                F.b = { { x[3], x[4], x[5] } };
                F.r = F.type == V2M_SDF_BOX ? 0 : x[6];
                F.r2 = F.type == V2M_SDF_CONE ? x[7] : F.r;
                F.L = norm(F.b - F.a);
                F.z = F.L > 0 ? unit(F.b - F.a) : F.z;
            } else if (F.type == V2M_SDF_SPHERE) {
                F.a = { { x[0], x[1], x[2] } };
                F.r = x[3];
            } else if (F.type == V2M_SDF_TORUS) {
                F.a = { { x[0], x[1], x[2] } };
                F.z = unit(A3{ { x[3], x[4], x[5] } });
                F.r = x[6];
                F.r2 = x[7];
            }

            PF.push_back(F);
        }

        lprog = sc.prog;
        const size_t N = static_cast<size_t>(lprog[0]);
        lprog[2] = lprog[3] = lprog[4] = 0.0f;
        lprog[5 + N] = 0.0f;
        lprog[6 + N] = 0.0f;
        lprog[8 + N] = 0.0f;
        st.primitives = static_cast<size_t>(np);

        for (int q = 0; q < np; ++q) {
            build_primitive(q);
        }

        st.patches = Pt.size();
        build_units();

        // the crossings: of each pair, marched on the first's patches, taken into the
        // second's (a patch on a plane with one of the second's: their crossing is that
        // one's sides, joined below)
        auto meet = [](const std::array<double, 6>& a, const std::array<double, 6>& b) {
            for (int k = 0; k < 3; ++k)
                if (a[static_cast<size_t>(k) + 3] < b[static_cast<size_t>(k)] || b[static_cast<size_t>(k) + 3] < a[static_cast<size_t>(k)]) {
                    return false;
                }

            return true;
        };

        for (int a = 0; a < np; ++a)
            for (int b = a + 1; b < np; ++b) {
                if (!meet(sc.pbox[static_cast<size_t>(a)], sc.pbox[static_cast<size_t>(b)])) {
                    continue;
                }

                const size_t c0 = C.size();

                for (size_t fi = 0; fi < Pt.size(); ++fi)
                    if (Pt[fi].prim == a && meet(Pt[fi].box, sc.pbox[static_cast<size_t>(b)]) && !shares_plane(static_cast<int>(fi), b)) {
                        march(static_cast<int>(fi), b);
                    }

                for (size_t ci = c0; ci < C.size(); ++ci) {
                    import(C[ci]);
                }
            }

        st.curves = C.size();
        thin_chains();

        if (std::getenv("V2M_CSG_DEBUG")) {
            for (const Chain& ch : C) {
                std::fprintf(stderr, "[csg-debug] chain %d-%d: %zu nodes, %s\n", ch.a, ch.b, ch.n.size(),
                             ch.n.front() == ch.n.back() ? "closed" : "open");
            }
        }

        merge_edges();
        junctions();
        side_splits();
        dedup();
        triples();
        thread_triples();
        Mesh m;

        for (const Unit& u : U) {
            tessellate(u, m.tris, m.tri_labels);
        }

        // on a box's face: exactly on its plane (a crossing's root, 1e-14 off it, would
        // leave the tets' faces there not quite flat)
        for (const Field& F : PF)
            if (F.type == V2M_SDF_BOX)
                for (size_t i = 0; i < X.size(); i += 3)
                    for (size_t k = 0; k < 3; ++k) {
                        if (std::fabs(X[i + k] - F.a[k]) <= near) {
                            X[i + k] = F.a[k];
                        } else if (std::fabs(X[i + k] - F.b[k]) <= near) {
                            X[i + k] = F.b[k];
                        }
                    }

        weld(m);
        m.nodes = X;
        compact_nodes(m);
        st.triangles = m.tris.size() / 3;
        st.nodes = m.nodes.size() / 3;
        return m;
    }
};

}  // namespace

Mesh csg_surface(const ShapeScene& sc, const CsgSurfOptions& o, CsgSurfStats* st) {
    Csg c(sc, o);
    Mesh m = c.run();

    if (st) {
        *st = c.st;
    }

    return m;
}

}  // namespace tn
