// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_surfgeom.h -- exact curve and surface geometry shared by the STEP reader
// (v2m_step.cpp) and the exact-surface CSG (v2m_csgsurf.cpp): small vectors,
// B-splines, curves (line, circle, ellipse, B-spline, polyline) and surfaces
// (plane, disk, cylinder, cone, sphere, torus, B-spline, revolution,
// extrusion) with their evaluation, inversion (a point -> its (u, v)) and
// normals. Header-only, inline.

#ifndef V2MESH_SURFGEOM_H
#define V2MESH_SURFGEOM_H

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace tn {
namespace sg {

const double kPi = 3.14159265358979323846;

// ---- small vectors --------------------------------------------------------------

typedef std::array<double, 3> A3;
inline A3 operator+(const A3& a, const A3& b) {
    return { { a[0] + b[0], a[1] + b[1], a[2] + b[2] } };
}
inline A3 operator-(const A3& a, const A3& b) {
    return { { a[0] - b[0], a[1] - b[1], a[2] - b[2] } };
}
inline A3 operator*(double s, const A3& a) {
    return { { s* a[0], s* a[1], s* a[2] } };
}
inline double dot(const A3& a, const A3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline A3 cross(const A3& a, const A3& b) {
    return { { a[1]* b[2] - a[2]* b[1], a[2]* b[0] - a[0]* b[2], a[0]* b[1] - a[1]* b[0] } };
}
inline double norm(const A3& a) {
    return std::sqrt(dot(a, a));
}
inline A3 unit(const A3& a) {
    const double n = norm(a);
    return n > 0 ? (1.0 / n) * a : A3{ { 0, 0, 0 } };
}
inline double wrap(double d, double P) {   // into (-P/2, P/2]
    d = std::fmod(d, P);

    if (d > 0.5 * P) {
        d -= P;
    } else if (d <= -0.5 * P) {
        d += P;
    }

    return d;
}

struct Frame {
    A3 o{ { 0, 0, 0 } }, x{ { 1, 0, 0 } }, y{ { 0, 1, 0 } }, z{ { 0, 0, 1 } };
};

// ---- B-splines ------------------------------------------------------------------

// the knot vector, multiplicities expanded
inline std::vector<double> expand_knots(const std::vector<double>& k, const std::vector<int>& m) {
    std::vector<double> r;

    for (size_t i = 0; i < k.size() && i < m.size(); ++i)
        for (int j = 0; j < m[i]; ++j) {
            r.push_back(k[i]);
        }

    return r;
}

// the span of t: knots[s] <= t < knots[s + 1], p <= s < n
inline int span_of(const std::vector<double>& U, int p, int n, double t) {
    if (t >= U[static_cast<size_t>(n)]) {
        return n - 1;
    }

    if (t <= U[static_cast<size_t>(p)]) {
        return p;
    }

    int lo = p, hi = n;

    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;

        if (t < U[static_cast<size_t>(mid)]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    return lo;
}

// the nonzero basis functions N[s-p .. s] at t
inline void basis(const std::vector<double>& U, int p, int s, double t, double* N) {
    double left[32], right[32];
    N[0] = 1;

    for (int j = 1; j <= p; ++j) {
        left[j] = t - U[static_cast<size_t>(s + 1 - j)];
        right[j] = U[static_cast<size_t>(s + j)] - t;
        double saved = 0;

        for (int r = 0; r < j; ++r) {
            const double den = right[r + 1] + left[j - r];
            const double tmp = den != 0 ? N[r] / den : 0;
            N[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }

        N[j] = saved;
    }
}

// ---- curves ---------------------------------------------------------------------

struct Curve {
    enum Kind { NONE, LINE, CIRCLE, ELLIPSE, BSPLINE, POLY } k = NONE;
    Frame f;
    A3 p{ { 0, 0, 0 } }, d{ { 1, 0, 0 } };   // a line: p + t d (d unit, t in mm)
    double r = 0, r2 = 0;                    // circle radius; ellipse semi-axes r, r2
    int deg = 0;                             // B-spline
    std::vector<A3> cp;                      // control points (a polyline: its points)
    std::vector<double> w, U;                // weights (empty: polynomial), knots
    bool closed = false;

    bool periodic() const {
        return k == CIRCLE || k == ELLIPSE;
    }
    void range(double& a, double& b) const {
        if (k == BSPLINE) {
            a = U[static_cast<size_t>(deg)];
            b = U[cp.size()];
        } else if (k == POLY) {
            a = 0;
            b = static_cast<double>(cp.size() - 1);
        } else if (periodic()) {
            a = 0;
            b = 2 * kPi;
        } else {
            a = -1e30;
            b = 1e30;
        }
    }
    A3 eval(double t) const {
        switch (k) {
            case LINE:
                return p + t * d;

            case CIRCLE:
                return f.o + r * (std::cos(t) * f.x + std::sin(t) * f.y);

            case ELLIPSE:
                return f.o + (r * std::cos(t)) * f.x + (r2 * std::sin(t)) * f.y;

            case POLY: {
                const int n = static_cast<int>(cp.size());
                t = std::min(static_cast<double>(n - 1), std::max(0.0, t));
                const int i = std::min(n - 2, std::max(0, static_cast<int>(std::floor(t))));
                const double s = t - i;
                return cp[static_cast<size_t>(i)] + s * (cp[static_cast<size_t>(i + 1)] - cp[static_cast<size_t>(i)]);
            }

            case BSPLINE: {
                const int n = static_cast<int>(cp.size());
                t = std::min(U[static_cast<size_t>(n)], std::max(U[static_cast<size_t>(deg)], t));
                const int s = span_of(U, deg, n, t);
                double N[32];
                basis(U, deg, s, t, N);
                A3 q{ { 0, 0, 0 } };
                double W = 0;

                for (int j = 0; j <= deg; ++j) {
                    const size_t c = static_cast<size_t>(s - deg + j);
                    const double wj = w.empty() ? 1.0 : w[c];
                    q = q + (N[j] * wj) * cp[c];
                    W += N[j] * wj;
                }

                return W != 0 ? (1.0 / W) * q : q;
            }

            default:
                return p;
        }
    }
    // the parameter of the curve's point nearest q
    double param(const A3& q) const {
        switch (k) {
            case LINE:
                return dot(q - p, d);

            case CIRCLE:
                return std::atan2(dot(q - f.o, f.y), dot(q - f.o, f.x));

            case ELLIPSE:
                return std::atan2(dot(q - f.o, f.y) / r2, dot(q - f.o, f.x) / r);

            default:
                break;
        }

        double a, b;
        range(a, b);
        const int ns = k == POLY ? static_cast<int>(cp.size() - 1) * 8 : std::max(64, 16 * static_cast<int>(cp.size()));
        double bt = a, bd = 1e300;

        for (int i = 0; i <= ns; ++i) {
            const double t = a + (b - a) * i / ns;
            const double dd = dot(eval(t) - q, eval(t) - q);

            if (dd < bd) {
                bd = dd;
                bt = t;
            }
        }

        // refine: golden-section in the bracket round it
        double lo = std::max(a, bt - (b - a) / ns), hi = std::min(b, bt + (b - a) / ns);

        for (int it = 0; it < 60; ++it) {
            const double m1 = lo + (hi - lo) * 0.381966, m2 = lo + (hi - lo) * 0.618034;
            const A3 e1 = eval(m1) - q, e2 = eval(m2) - q;

            if (dot(e1, e1) < dot(e2, e2)) {
                hi = m2;
            } else {
                lo = m1;
            }
        }

        return 0.5 * (lo + hi);
    }
};

// ---- surfaces -------------------------------------------------------------------

struct Surf {
    enum Kind { NONE, PLANE, CYL, CONE, SPHERE, TORUS, BSPLINE, REVOL, EXTRU, DISK } k = NONE;
    Frame f;
    double r = 0, R = 0, ang = 0;            // radius; torus major R; cone semi-angle
    int du = 0, dv = 0, nu = 0, nv = 0;      // B-spline degrees, control net nu x nv
    std::vector<A3> cp;                      // [iu * nv + iv]
    std::vector<double> w, Uk, Vk;
    std::shared_ptr<Curve> c;                // revolution / extrusion: the swept curve
    A3 axis_o{ { 0, 0, 0 } }, axis_d{ { 0, 0, 1 } }, ext{ { 0, 0, 1 } };
    // the domain (B-spline, revolution, extrusion); a periodic direction's period
    double u0 = -1e30, u1 = 1e30, v0 = -1e30, v1 = 1e30;
    bool per_u = false, per_v = false;
    double stol = 1e-9;   // (mm) how near a pole / an apex counts as on it
    mutable std::vector<std::array<double, 5>> grid;   // (u, v, x, y, z) samples for inversion

    A3 eval(double u, double v) const {
        switch (k) {
            case PLANE:
                return f.o + u * f.x + v * f.y;

            case CYL:
                return f.o + r * (std::cos(u) * f.x + std::sin(u) * f.y) + v * f.z;

            case DISK:   // (u the angle, v the radius)
                return f.o + v * (std::cos(u) * f.x + std::sin(u) * f.y);

            case CONE:
                return f.o + (r + v * std::tan(ang)) * (std::cos(u) * f.x + std::sin(u) * f.y) + v * f.z;

            case SPHERE:
                return f.o + (r * std::cos(v)) * (std::cos(u) * f.x + std::sin(u) * f.y) + (r * std::sin(v)) * f.z;

            case TORUS:
                return f.o + (R + r * std::cos(v)) * (std::cos(u) * f.x + std::sin(u) * f.y) + (r * std::sin(v)) * f.z;

            case BSPLINE: {
                // (a closed direction: into its knot range by whole periods; else clamped)
                if (per_u && u1 > u0) {
                    u = u0 + std::fmod(std::fmod(u - u0, u1 - u0) + (u1 - u0), u1 - u0);
                } else {
                    u = std::min(u1, std::max(u0, u));
                }

                if (per_v && v1 > v0) {
                    v = v0 + std::fmod(std::fmod(v - v0, v1 - v0) + (v1 - v0), v1 - v0);
                } else {
                    v = std::min(v1, std::max(v0, v));
                }

                const int su = span_of(Uk, du, nu, u), sv = span_of(Vk, dv, nv, v);
                double Nu[32], Nv[32];
                basis(Uk, du, su, u, Nu);
                basis(Vk, dv, sv, v, Nv);
                A3 q{ { 0, 0, 0 } };
                double W = 0;

                for (int i = 0; i <= du; ++i)
                    for (int j = 0; j <= dv; ++j) {
                        const size_t id = static_cast<size_t>((su - du + i) * nv + (sv - dv + j));
                        const double wij = (w.empty() ? 1.0 : w[id]) * Nu[i] * Nv[j];
                        q = q + wij * cp[id];
                        W += wij;
                    }

                return W != 0 ? (1.0 / W) * q : q;
            }

            case REVOL: {   // the curve's point at v, turned by u about the axis
                const A3 q = c->eval(v) - axis_o;
                const A3 a = axis_d;
                const double cu = std::cos(u), s = std::sin(u);
                return axis_o + (cu * q) + (s * cross(a, q)) + ((1 - cu) * dot(a, q)) * a;
            }

            case EXTRU:
                return c->eval(u) + v * ext;

            default:
                return f.o;
        }
    }
    // a singular (u-independent) point of the parameterisation at v
    bool singular_v(double& vs) const {
        if (k == CONE && std::tan(ang) != 0) {
            vs = -r / std::tan(ang);
            return true;
        }

        return false;
    }
    void inv(const A3& p, double& u, double& v, bool& sing, double hu, double hv, bool hint) const {
        sing = false;
        const A3 q = p - f.o;
        const double X = dot(q, f.x), Y = dot(q, f.y), Z = dot(q, f.z);

        switch (k) {
            case PLANE:
                u = X;
                v = Y;
                return;

            case CYL:
                u = std::atan2(Y, X);
                v = Z;
                return;

            case DISK:
                v = std::hypot(X, Y);
                sing = v <= stol;
                u = std::atan2(Y, X);
                return;

            case CONE: {
                v = Z;
                const double rad = r + v * std::tan(ang);
                sing = std::fabs(rad) <= stol;
                u = std::atan2(Y, X);

                if (rad < 0) {
                    u += kPi;   // (the far nappe)
                }

                return;
            }

            case SPHERE: {
                const double rho = std::hypot(X, Y);
                sing = rho <= stol;
                u = std::atan2(Y, X);
                v = std::atan2(Z, rho);
                return;
            }

            case TORUS: {
                // (a spindle torus, minor > major: a point on its inner part, R + r cos v < 0,
                // is at u + pi, from the tube's centre on the far side)
                const double rho = std::hypot(X, Y);
                const double d1 = std::hypot(rho - R, Z), d2 = std::hypot(rho + R, Z);

                if (r > R && std::fabs(d2 - r) < std::fabs(d1 - r)) {
                    u = std::atan2(Y, X) + kPi;
                    v = std::atan2(Z, -rho - R);
                } else {
                    u = std::atan2(Y, X);
                    v = std::atan2(Z, rho - R);
                }

                return;
            }

            case EXTRU: {   // along the extrusion onto the curve, then its parameter
                double a0, a1;
                c->range(a0, a1);
                const A3 base = c->eval(c->k == Curve::LINE ? 0.0 : a0);
                const A3 pp = p - dot(p - base, ext) * ext;   // (the curve's plane through base)
                u = c->param(pp);
                v = dot(p - c->eval(u), ext);
                return;
            }

            case REVOL: {   // the profile point at p's height and radius about the axis
                const A3 qa = p - axis_o;
                const double h = dot(qa, axis_d);
                const A3 rv = qa - h * axis_d;
                const double rho = norm(rv);
                auto miss = [&](double t) {
                    const A3 cq = c->eval(t) - axis_o;
                    const double ch = dot(cq, axis_d);
                    const double cr = norm(cq - ch * axis_d);
                    return (ch - h) * (ch - h) + (cr - rho) * (cr - rho);
                };
                const int ns = c->k == Curve::BSPLINE ? std::max(64, 16 * static_cast<int>(c->cp.size())) : 256;
                double bt = v0, bd = 1e300;

                for (int i = 0; i <= ns; ++i) {
                    const double t = v0 + (v1 - v0) * i / ns, d2 = miss(t);

                    if (d2 < bd) {
                        bd = d2;
                        bt = t;
                    }
                }

                double lo = std::max(v0, bt - (v1 - v0) / ns), hi = std::min(v1, bt + (v1 - v0) / ns);

                for (int it = 0; it < 80; ++it) {
                    const double m1 = lo + (hi - lo) * 0.381966, m2 = lo + (hi - lo) * 0.618034;

                    if (miss(m1) < miss(m2)) {
                        hi = m2;
                    } else {
                        lo = m1;
                    }
                }

                v = 0.5 * (lo + hi);
                const A3 cq = c->eval(v) - axis_o;
                const A3 cr = cq - dot(cq, axis_d) * axis_d;
                sing = rho <= stol || norm(cr) <= stol;
                u = sing ? 0 : std::atan2(dot(cross(cr, rv), axis_d), dot(cr, rv));
                return;
            }

            default:
                break;
        }

        // numerical: the nearest grid sample (or the hint), then Newton on |S - p|^2
        if (grid.empty()) {
            const int n = 32;

            for (int i = 0; i <= n; ++i)
                for (int j = 0; j <= n; ++j) {
                    const double uu = u0 + (u1 - u0) * i / n, vv = v0 + (v1 - v0) * j / n;
                    const A3 s = eval(uu, vv);
                    grid.push_back({ { uu, vv, s[0], s[1], s[2] } });
                }
        }

        auto newton = [&](double & uu, double & vv) {
            const double eu = 1e-7 * std::max(1.0, u1 - u0), ev = 1e-7 * std::max(1.0, v1 - v0);

            for (int it = 0; it < 30; ++it) {
                const A3 s = eval(uu, vv), e = s - p;
                const A3 su = (1.0 / (2 * eu)) * (eval(uu + eu, vv) - eval(uu - eu, vv));
                const A3 sv = (1.0 / (2 * ev)) * (eval(uu, vv + ev) - eval(uu, vv - ev));
                const double a = dot(su, su), b = dot(su, sv), cc = dot(sv, sv), g1 = dot(su, e), g2 = dot(sv, e);
                const double det = a * cc - b * b;

                if (!(std::fabs(det) > 1e-300)) {
                    break;
                }

                const double dU = (cc * g1 - b * g2) / det, dV = (a * g2 - b * g1) / det;
                uu -= dU;
                vv -= dV;

                if (!per_u) {
                    uu = std::min(u1, std::max(u0, uu));
                }

                if (!per_v) {
                    vv = std::min(v1, std::max(v0, vv));
                }

                if (std::fabs(dU) < 1e-12 * std::max(1.0, u1 - u0) && std::fabs(dV) < 1e-12 * std::max(1.0, v1 - v0)) {
                    break;
                }
            }

            const A3 e = eval(uu, vv) - p;
            return dot(e, e);
        };
        double bu = 0, bv = 0, bd = 1e300;

        for (const auto& g : grid) {
            const double dd = (g[2] - p[0]) * (g[2] - p[0]) + (g[3] - p[1]) * (g[3] - p[1]) + (g[4] - p[2]) * (g[4] - p[2]);

            if (dd < bd) {
                bd = dd;
                bu = g[0];
                bv = g[1];
            }
        }

        bd = newton(bu, bv);

        if (hint) {
            double uu = hu, vv = hv;
            const double dd = newton(uu, vv);

            if (dd <= bd * (1 + 1e-9) + 1e-24) {
                bu = uu;
                bv = vv;
            }
        }

        u = bu;
        v = bv;
    }
    A3 normal(double u, double v) const {
        const double e = 1e-6;
        const A3 su = eval(u + e, v) - eval(u - e, v), sv = eval(u, v + e) - eval(u, v - e);
        return unit(cross(su, sv));
    }
    double period_u() const {
        return per_u ? (k == BSPLINE || k == EXTRU ? u1 - u0 : 2 * kPi) : 0;
    }
    double period_v() const {
        return per_v ? (k == BSPLINE ? v1 - v0 : 2 * kPi) : 0;
    }
};


}  // namespace sg
}  // namespace tn

#endif  // V2MESH_SURFGEOM_H
