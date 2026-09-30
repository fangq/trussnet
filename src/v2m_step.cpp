// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_step.cpp -- see v2m_step.h.

#include "v2m_step.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "v2m_modes.h"  // compact_nodes
#include "v2m_plc.h"    // cdt2d

namespace tn {

namespace {

const double kPi = 3.14159265358979323846;

// ---- the file: ISO 10303-21 -----------------------------------------------------

struct Val {
    enum Kind { NUL, STAR, NUM, STR, ENUM, REF, LIST, TYPED } k = NUL;
    double x = 0;
    long r = 0;
    std::string s;          // a string, an enum's name, a typed value's type
    std::vector<Val> l;     // a list, a typed value's arguments
};

struct Ent {
    std::string type;                                          // simple: its type
    std::vector<Val> a;                                        // simple: its arguments
    std::vector<std::pair<std::string, std::vector<Val>>> parts;   // complex: (type, arguments) each
};

struct Parser {
    const std::string& t;
    size_t i = 0;
    std::string path;

    Parser(const std::string& text, const std::string& p) : t(text), path(p) {}

    [[noreturn]] void fail(const std::string& what) const {
        size_t line = 1;

        for (size_t k = 0; k < i && k < t.size(); ++k) {
            line += t[k] == '\n';
        }

        throw std::runtime_error(path + ": line " + std::to_string(line) + ": " + what);
    }
    void ws() {
        for (;;) {
            while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) {
                ++i;
            }

            if (i + 1 < t.size() && t[i] == '/' && t[i + 1] == '*') {
                const size_t e = t.find("*/", i + 2);
                i = e == std::string::npos ? t.size() : e + 2;
                continue;
            }

            break;
        }
    }
    bool eat(char c) {
        ws();

        if (i < t.size() && t[i] == c) {
            ++i;
            return true;
        }

        return false;
    }
    void need(char c) {
        if (!eat(c)) {
            fail(std::string("expected '") + c + "'");
        }
    }
    std::string ident() {
        ws();
        const size_t s = i;

        while (i < t.size() && (std::isalnum(static_cast<unsigned char>(t[i])) || t[i] == '_' || t[i] == '-')) {
            ++i;
        }

        if (i == s) {
            fail("expected a name");
        }

        std::string r = t.substr(s, i - s);
        std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return r;
    }
    Val value() {
        ws();
        Val v;

        if (i >= t.size()) {
            fail("unexpected end");
        }

        const char c = t[i];

        if (c == '$') {
            ++i;
        } else if (c == '*') {
            ++i;
            v.k = Val::STAR;
        } else if (c == '\'') {   // a string ('' an escaped quote)
            ++i;
            v.k = Val::STR;

            for (;;) {
                if (i >= t.size()) {
                    fail("an unterminated string");
                }

                if (t[i] == '\'') {
                    if (i + 1 < t.size() && t[i + 1] == '\'') {
                        v.s += '\'';
                        i += 2;
                        continue;
                    }

                    ++i;
                    break;
                }

                v.s += t[i++];
            }
        } else if (c == '"') {   // a binary
            const size_t e = t.find('"', i + 1);
            i = e == std::string::npos ? t.size() : e + 1;
            v.k = Val::STR;
        } else if (c == '.') {   // an enumeration (or .T. / .F.)
            ++i;
            const size_t e = t.find('.', i);

            if (e == std::string::npos) {
                fail("an unterminated enumeration");
            }

            v.k = Val::ENUM;
            v.s = t.substr(i, e - i);
            i = e + 1;
        } else if (c == '#') {
            ++i;
            const size_t s = i;

            while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) {
                ++i;
            }

            v.k = Val::REF;
            v.r = std::strtol(t.substr(s, i - s).c_str(), nullptr, 10);
        } else if (c == '(') {
            ++i;
            v.k = Val::LIST;
            ws();

            if (!eat(')')) {
                for (;;) {
                    v.l.push_back(value());

                    if (eat(')')) {
                        break;
                    }

                    need(',');
                }
            }
        } else if (c == '+' || c == '-' || c == '.' || std::isdigit(static_cast<unsigned char>(c))) {
            char* e = nullptr;
            v.k = Val::NUM;
            v.x = std::strtod(t.c_str() + i, &e);

            if (e == t.c_str() + i) {
                fail("a bad number");
            }

            i = static_cast<size_t>(e - t.c_str());
        } else if (std::isalpha(static_cast<unsigned char>(c))) {   // a typed value NAME(..)
            v.k = Val::TYPED;
            v.s = ident();
            Val a = value();

            if (a.k == Val::LIST) {
                v.l = std::move(a.l);
            } else {
                v.l.push_back(std::move(a));
            }
        } else {
            fail(std::string("unexpected '") + c + "'");
        }

        return v;
    }
    std::vector<Val> args() {
        Val v = value();

        if (v.k != Val::LIST) {
            fail("expected an argument list");
        }

        return std::move(v.l);
    }
};

void parse_step(const std::string& path, std::unordered_map<long, Ent>& E) {
    std::ifstream f(path, std::ios::binary);

    if (!f) {
        throw std::runtime_error(path + ": cannot open");
    }

    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    if (text.find("ISO-10303-21") == std::string::npos) {
        throw std::runtime_error(path + ": not a STEP file (no ISO-10303-21 header)");
    }

    Parser P(text, path);

    for (size_t d = text.find("DATA"); d != std::string::npos; d = text.find("DATA", d + 4)) {
        P.i = d + 4;

        // DATA; or DATA(..);
        if (P.eat('(')) {
            int depth = 1;

            while (P.i < text.size() && depth > 0) {
                depth += text[P.i] == '(' ? 1 : text[P.i] == ')' ? -1 : 0;
                ++P.i;
            }
        }

        if (!P.eat(';')) {
            continue;   // (the word elsewhere)
        }

        for (;;) {
            P.ws();

            if (P.i >= text.size()) {
                break;
            }

            if (text.compare(P.i, 6, "ENDSEC") == 0) {
                P.i += 6;
                break;
            }

            P.need('#');
            const size_t s = P.i;

            while (P.i < text.size() && std::isdigit(static_cast<unsigned char>(text[P.i]))) {
                ++P.i;
            }

            const long id = std::strtol(text.substr(s, P.i - s).c_str(), nullptr, 10);
            P.need('=');
            Ent e;

            if (P.eat('(')) {   // complex: (A(..) B(..) ..)
                while (!P.eat(')')) {
                    std::string nm = P.ident();
                    e.parts.emplace_back(nm, P.args());
                }
            } else {
                e.type = P.ident();
                e.a = P.args();
            }

            P.need(';');
            E[id] = std::move(e);
        }
    }

    if (E.empty()) {
        throw std::runtime_error(path + ": no DATA section");
    }
}

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
std::vector<double> expand_knots(const std::vector<double>& k, const std::vector<int>& m) {
    std::vector<double> r;

    for (size_t i = 0; i < k.size() && i < m.size(); ++i)
        for (int j = 0; j < m[i]; ++j) {
            r.push_back(k[i]);
        }

    return r;
}

// the span of t: knots[s] <= t < knots[s + 1], p <= s < n
int span_of(const std::vector<double>& U, int p, int n, double t) {
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
void basis(const std::vector<double>& U, int p, int s, double t, double* N) {
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
    enum Kind { NONE, PLANE, CYL, CONE, SPHERE, TORUS, BSPLINE, REVOL, EXTRU } k = NONE;
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

// ---- the model: entities -> geometry and topology -------------------------------

struct Model {
    std::unordered_map<long, Ent> E;
    double len = 1, angle = 1;   // file units -> mm, rad
    double diag = 1;             // the model's size (mm): a line's parameter range
    std::string path;
    std::vector<std::string> notes;
    size_t note_max = 12;

    void note(const std::string& s) {
        if (notes.size() < note_max && std::find(notes.begin(), notes.end(), s) == notes.end()) {
            notes.push_back(s);
        }
    }
    const Ent& ent(long id) const {
        const auto it = E.find(id);

        if (it == E.end()) {
            throw std::runtime_error(path + ": #" + std::to_string(id) + " is not defined");
        }

        return it->second;
    }
    const Ent& ent(const Val& v) const {
        if (v.k != Val::REF) {
            throw std::runtime_error(path + ": expected a reference");
        }

        return ent(v.r);
    }
    // the arguments of an entity's part `type` (a simple entity of that type: its own)
    static const std::vector<Val>* part(const Ent& e, const char* type) {
        if (e.type == type) {
            return &e.a;
        }

        for (const auto& p : e.parts)
            if (p.first == type) {
                return &p.second;
            }

        return nullptr;
    }
    static bool is(const Ent& e, const char* type) {
        return part(e, type) != nullptr;
    }
    static double num(const Val& v) {
        if (v.k == Val::NUM) {
            return v.x;
        }

        if (v.k == Val::TYPED && !v.l.empty()) {
            return num(v.l[0]);
        }

        return 0;
    }
    static const std::vector<Val>& list(const Val& v) {
        static const std::vector<Val> none;
        return v.k == Val::LIST ? v.l : none;
    }
    static bool flag(const Val& v) {
        return v.k == Val::ENUM && (v.s == "T" || v.s == "TRUE");
    }

    A3 point(const Val& v) const {   // CARTESIAN_POINT, in mm
        const Ent& e = ent(v);
        const auto* a = part(e, "CARTESIAN_POINT");

        if (!a || a->size() < 2) {
            throw std::runtime_error(path + ": expected a CARTESIAN_POINT");
        }

        const auto& c = list((*a)[1]);
        A3 p{ { 0, 0, 0 } };

        for (size_t i = 0; i < 3 && i < c.size(); ++i) {
            p[i] = len * num(c[i]);
        }

        return p;
    }
    A3 direction(const Val& v) const {
        const Ent& e = ent(v);
        const auto* a = part(e, "DIRECTION");

        if (a && a->size() >= 2) {
            const auto& c = list((*a)[1]);
            A3 d{ { 0, 0, 0 } };

            for (size_t i = 0; i < 3 && i < c.size(); ++i) {
                d[i] = num(c[i]);
            }

            return unit(d);
        }

        const auto* vec = part(e, "VECTOR");   // VECTOR(name, orientation, magnitude)

        if (vec && vec->size() >= 3) {
            return direction((*vec)[1]);
        }

        throw std::runtime_error(path + ": expected a DIRECTION");
    }
    Frame placement(const Val& v) const {   // AXIS2_PLACEMENT_3D(name, location, axis, ref)
        const Ent& e = ent(v);
        Frame f;
        const auto* a = part(e, "AXIS2_PLACEMENT_3D");

        if (!a || a->size() < 2) {
            throw std::runtime_error(path + ": expected an AXIS2_PLACEMENT_3D");
        }

        f.o = point((*a)[1]);

        if (a->size() > 2 && (*a)[2].k == Val::REF) {
            f.z = direction((*a)[2]);
        }

        A3 x{ { 1, 0, 0 } };

        if (a->size() > 3 && (*a)[3].k == Val::REF) {
            x = direction((*a)[3]);
        } else if (std::fabs(f.z[0]) > 0.9) {
            x = { { 0, 1, 0 } };
        }

        f.x = unit(x - dot(x, f.z) * f.z);

        if (norm(f.x) == 0) {
            f.x = unit(cross(f.z, std::fabs(f.z[0]) > 0.9 ? A3{ { 0, 1, 0 } } : A3{ { 1, 0, 0 } }));
        }

        f.y = cross(f.z, f.x);
        return f;
    }

    // the length and angle units (the first global unit context's)
    void units() {
        for (const auto& kv : E) {
            const auto* g = part(kv.second, "GLOBAL_UNIT_ASSIGNED_CONTEXT");

            if (!g || g->empty()) {
                continue;
            }

            for (const Val& u : list((*g)[0])) {
                if (u.k != Val::REF) {
                    continue;
                }

                const Ent& e = ent(u);

                if (is(e, "LENGTH_UNIT")) {
                    len = unit_scale(e, true);
                } else if (is(e, "PLANE_ANGLE_UNIT")) {
                    angle = unit_scale(e, false);
                }
            }

            break;
        }
    }
    double unit_scale(const Ent& e, bool length) const {   // mm (length) or rad (angle) per file unit
        const auto* si = part(e, "SI_UNIT");

        if (si && si->size() >= 2) {
            double f = 1;
            const Val& pre = (*si)[0];

            if (pre.k == Val::ENUM) {
                static const std::map<std::string, double> P = { { "EXA", 1e18 }, { "PETA", 1e15 }, { "TERA", 1e12 },
                    { "GIGA", 1e9 }, { "MEGA", 1e6 }, { "KILO", 1e3 }, { "HECTO", 1e2 }, { "DECA", 1e1 }, { "DECI", 1e-1 },
                    { "CENTI", 1e-2 }, { "MILLI", 1e-3 }, { "MICRO", 1e-6 }, { "NANO", 1e-9 }, { "PICO", 1e-12 }
                };
                const auto it = P.find(pre.s);

                if (it != P.end()) {
                    f = it->second;
                }
            }

            return length ? 1000.0 * f : f;
        }

        const auto* cb = part(e, "CONVERSION_BASED_UNIT");   // (name, measure_with_unit)

        if (cb && cb->size() >= 2 && (*cb)[1].k == Val::REF) {
            const Ent& m = ent((*cb)[1]);
            const std::vector<Val>* mw = nullptr;

            for (const char* t : {
                        "LENGTH_MEASURE_WITH_UNIT", "PLANE_ANGLE_MEASURE_WITH_UNIT", "MEASURE_WITH_UNIT"
                    })

                if (!mw) {
                    mw = part(m, t);
                }

            if (mw && mw->size() >= 2 && (*mw)[1].k == Val::REF) {
                return num((*mw)[0]) * unit_scale(ent((*mw)[1]), length);
            }
        }

        return length ? 1.0 : 1.0;
    }

    std::unordered_map<long, std::shared_ptr<Curve>> curves;
    std::unordered_map<long, std::shared_ptr<Surf>> surfs;

    std::shared_ptr<Curve> curve(long id) {
        const auto it = curves.find(id);

        if (it != curves.end()) {
            return it->second;
        }

        auto c = std::make_shared<Curve>();
        const Ent& e = ent(id);
        const std::vector<Val>* a;

        if ((a = part(e, "SURFACE_CURVE")) || (a = part(e, "SEAM_CURVE")) || (a = part(e, "INTERSECTION_CURVE"))) {
            c = curve((*a)[1].r);   // (name, curve_3d, (pcurves), master)
        } else if ((a = part(e, "TRIMMED_CURVE"))) {
            c = curve((*a)[1].r);   // (the edge's vertices trim it)
        } else if ((a = part(e, "LINE"))) {   // (name, pnt, VECTOR)
            c->k = Curve::LINE;
            c->p = point((*a)[1]);
            c->d = direction((*a)[2]);
        } else if ((a = part(e, "CIRCLE"))) {   // (name, placement, radius)
            c->k = Curve::CIRCLE;
            c->f = placement((*a)[1]);
            c->r = len * num((*a)[2]);
        } else if ((a = part(e, "ELLIPSE"))) {   // (name, placement, semi1, semi2)
            c->k = Curve::ELLIPSE;
            c->f = placement((*a)[1]);
            c->r = len * num((*a)[2]);
            c->r2 = len * num((*a)[3]);
        } else if ((a = part(e, "POLYLINE"))) {
            c->k = Curve::POLY;

            for (const Val& p : list((*a)[1])) {
                c->cp.push_back(point(p));
            }
        } else if (is(e, "B_SPLINE_CURVE") || e.type == "B_SPLINE_CURVE_WITH_KNOTS") {
            c->k = Curve::BSPLINE;
        } else {
            note("curve " + (e.type.empty() ? std::string("(complex)") : e.type) + " not supported");
            c->k = Curve::NONE;
        }

        if (c->k == Curve::BSPLINE && c->cp.empty()) {
            bspline_curve(e, *c);
        }

        curves[id] = c;
        return c;
    }
    void bspline_curve(const Ent& e, Curve& c) {
        const std::vector<Val>* b = part(e, "B_SPLINE_CURVE");
        const std::vector<Val>* k = part(e, "B_SPLINE_CURVE_WITH_KNOTS");
        std::vector<Val> deg_cps;   // deg, cps, form, closed, self
        std::vector<Val> knots;     // mults, knots, spec

        if (e.type == "B_SPLINE_CURVE_WITH_KNOTS") {   // (name, deg, cps, form, closed, self, mults, knots, spec)
            deg_cps.assign(e.a.begin() + 1, e.a.begin() + 6);
            knots.assign(e.a.begin() + 6, e.a.end());
        } else if (b && k) {
            deg_cps = *b;
            knots = *k;

            if (deg_cps.size() > 5) {
                deg_cps.erase(deg_cps.begin());   // (a name first)
            }
        } else {
            note("B-spline curve without knots (uniform / Bezier) not supported");
            c.k = Curve::NONE;
            return;
        }

        c.deg = static_cast<int>(num(deg_cps[0]));

        for (const Val& p : list(deg_cps[1])) {
            c.cp.push_back(point(p));
        }

        c.closed = deg_cps.size() > 3 && flag(deg_cps[3]);
        std::vector<int> m;
        std::vector<double> kv;

        for (const Val& v : list(knots[0])) {
            m.push_back(static_cast<int>(num(v)));
        }

        for (const Val& v : list(knots[1])) {
            kv.push_back(num(v));
        }

        c.U = expand_knots(kv, m);

        if (const auto* rw = part(e, "RATIONAL_B_SPLINE_CURVE")) {
            for (const Val& v : list((*rw)[0])) {
                c.w.push_back(num(v));
            }
        }

        if (c.deg < 1 || c.deg > 30 || c.U.size() != c.cp.size() + static_cast<size_t>(c.deg) + 1 ||
                (!c.w.empty() && c.w.size() != c.cp.size())) {
            note("a malformed B-spline curve");
            c.k = Curve::NONE;
        }
    }

    std::shared_ptr<Surf> surf(long id) {
        const auto it = surfs.find(id);

        if (it != surfs.end()) {
            return it->second;
        }

        auto s = std::make_shared<Surf>();
        const Ent& e = ent(id);
        const std::vector<Val>* a;

        if ((a = part(e, "PLANE"))) {
            s->k = Surf::PLANE;
            s->f = placement((*a)[1]);
        } else if ((a = part(e, "CYLINDRICAL_SURFACE"))) {
            s->k = Surf::CYL;
            s->f = placement((*a)[1]);
            s->r = len * num((*a)[2]);
            s->per_u = true;
        } else if ((a = part(e, "CONICAL_SURFACE"))) {
            s->k = Surf::CONE;
            s->f = placement((*a)[1]);
            s->r = len * num((*a)[2]);
            s->ang = angle * num((*a)[3]);
            s->per_u = true;
        } else if ((a = part(e, "SPHERICAL_SURFACE"))) {
            s->k = Surf::SPHERE;
            s->f = placement((*a)[1]);
            s->r = len * num((*a)[2]);
            s->per_u = true;
            s->v0 = -0.5 * kPi;
            s->v1 = 0.5 * kPi;
        } else if ((a = part(e, "TOROIDAL_SURFACE"))) {
            s->k = Surf::TORUS;
            s->f = placement((*a)[1]);
            s->R = len * num((*a)[2]);
            s->r = len * num((*a)[3]);
            s->per_u = s->per_v = true;
        } else if ((a = part(e, "SURFACE_OF_REVOLUTION"))) {   // (name, curve, AXIS1_PLACEMENT)
            s->k = Surf::REVOL;
            s->c = curve((*a)[1].r);
            const auto* ax = part(ent((*a)[2]), "AXIS1_PLACEMENT");

            if (!ax) {
                throw std::runtime_error(path + ": a SURFACE_OF_REVOLUTION without an AXIS1_PLACEMENT");
            }

            s->axis_o = point((*ax)[1]);

            if (ax->size() > 2 && (*ax)[2].k == Val::REF) {
                s->axis_d = direction((*ax)[2]);
            }

            s->per_u = true;
            s->u0 = 0;
            s->u1 = 2 * kPi;
            s->c->range(s->v0, s->v1);
        } else if ((a = part(e, "SURFACE_OF_LINEAR_EXTRUSION"))) {   // (name, curve, VECTOR)
            s->k = Surf::EXTRU;
            s->c = curve((*a)[1].r);
            s->ext = direction((*a)[2]);

            s->c->range(s->u0, s->u1);

            if (s->c->k == Curve::LINE) {
                s->u0 = -4 * diag;
                s->u1 = 4 * diag;
            }

            s->v0 = -4 * diag;
            s->v1 = 4 * diag;
        } else if (is(e, "B_SPLINE_SURFACE") || e.type == "B_SPLINE_SURFACE_WITH_KNOTS") {
            bspline_surf(e, *s);
        } else {
            note("surface " + (e.type.empty() ? std::string("(complex)") : e.type) + " not supported");
        }

        if (s->k == Surf::REVOL || s->k == Surf::EXTRU) {
            if (!s->c || s->c->k == Curve::NONE) {
                s->k = Surf::NONE;
            } else if (s->c->periodic() && s->k == Surf::EXTRU) {
                s->per_u = true;
                s->u0 = 0;
                s->u1 = 2 * kPi;
            }

            if (s->k == Surf::REVOL && s->c->periodic()) {
                s->per_v = true;
                s->v0 = 0;
                s->v1 = 2 * kPi;
            }

            if (s->k == Surf::REVOL && s->c->k == Curve::LINE) {
                s->v0 = -4 * diag;
                s->v1 = 4 * diag;
            }
        }

        s->stol = 1e-6 * diag;
        surfs[id] = s;
        return s;
    }
    void bspline_surf(const Ent& e, Surf& s) {
        std::vector<Val> head, knots;   // (du, dv, cps, form, uclosed, vclosed, self), (umults, vmults, uknots, vknots, spec)

        if (e.type == "B_SPLINE_SURFACE_WITH_KNOTS") {
            head.assign(e.a.begin() + 1, e.a.begin() + 8);
            knots.assign(e.a.begin() + 8, e.a.end());
        } else {
            const auto* b = part(e, "B_SPLINE_SURFACE");
            const auto* k = part(e, "B_SPLINE_SURFACE_WITH_KNOTS");

            if (!b || !k) {
                note("B-spline surface without knots (uniform / Bezier) not supported");
                return;
            }

            head = *b;
            knots = *k;

            if (head.size() > 7) {
                head.erase(head.begin());
            }
        }

        s.du = static_cast<int>(num(head[0]));
        s.dv = static_cast<int>(num(head[1]));
        const auto& rows = list(head[2]);
        s.nu = static_cast<int>(rows.size());
        s.nv = s.nu > 0 ? static_cast<int>(list(rows[0]).size()) : 0;

        for (const Val& row : rows) {
            const auto& r = list(row);

            if (static_cast<int>(r.size()) != s.nv) {
                note("a ragged B-spline control net");
                return;
            }

            for (const Val& p : r) {
                s.cp.push_back(point(p));
            }
        }

        std::vector<int> mu, mv;
        std::vector<double> ku, kv;

        for (const Val& v : list(knots[0])) {
            mu.push_back(static_cast<int>(num(v)));
        }

        for (const Val& v : list(knots[1])) {
            mv.push_back(static_cast<int>(num(v)));
        }

        for (const Val& v : list(knots[2])) {
            ku.push_back(num(v));
        }

        for (const Val& v : list(knots[3])) {
            kv.push_back(num(v));
        }

        s.Uk = expand_knots(ku, mu);
        s.Vk = expand_knots(kv, mv);

        if (const auto* rw = part(e, "RATIONAL_B_SPLINE_SURFACE")) {
            for (const Val& row : list((*rw)[0]))
                for (const Val& v : list(row)) {
                    s.w.push_back(num(v));
                }
        }

        if (s.du < 1 || s.dv < 1 || s.du > 30 || s.dv > 30 || s.Uk.size() != static_cast<size_t>(s.nu + s.du + 1) ||
                s.Vk.size() != static_cast<size_t>(s.nv + s.dv + 1) || (!s.w.empty() && s.w.size() != s.cp.size())) {
            note("a malformed B-spline surface");
            return;
        }

        s.k = Surf::BSPLINE;
        s.u0 = s.Uk[static_cast<size_t>(s.du)];
        s.u1 = s.Uk[static_cast<size_t>(s.nu)];
        s.v0 = s.Vk[static_cast<size_t>(s.dv)];
        s.v1 = s.Vk[static_cast<size_t>(s.nv)];
        // closed: the first and last rows / columns coincide
        auto same = [&](const A3 & p, const A3 & q) {
            return norm(p - q) <= 1e-9 * std::max(1.0, norm(p));
        };
        bool cu = true, cv = true;

        for (int j = 0; j < s.nv && cu; ++j) {
            cu = same(s.eval(s.u0, s.v0 + (s.v1 - s.v0) * j / std::max(1, s.nv - 1)),
                      s.eval(s.u1, s.v0 + (s.v1 - s.v0) * j / std::max(1, s.nv - 1)));
        }

        for (int i = 0; i < s.nu && cv; ++i) {
            cv = same(s.eval(s.u0 + (s.u1 - s.u0) * i / std::max(1, s.nu - 1), s.v0),
                      s.eval(s.u0 + (s.u1 - s.u0) * i / std::max(1, s.nu - 1), s.v1));
        }

        s.per_u = cu;
        s.per_v = cv;
    }
};

// ---- the tessellation -----------------------------------------------------------

struct Builder {
    Model& M;
    const StepOptions& o;
    double tol = 0, amax = 0, hmax = 0;
    std::vector<double> nodes;                       // mm
    std::vector<int32_t> tris, tlab;
    std::unordered_map<long, int32_t> vnode;         // VERTEX_POINT -> node
    std::unordered_map<long, std::vector<int32_t>> edges;   // EDGE_CURVE -> its nodes, start to end
    size_t faces = 0, failed = 0;

    Builder(Model& m, const StepOptions& op) : M(m), o(op) {}

    int32_t add(const A3& p) {
        nodes.insert(nodes.end(), p.begin(), p.end());
        return static_cast<int32_t>(nodes.size() / 3 - 1);
    }
    A3 at(int32_t n) const {
        return { { nodes[3 * static_cast<size_t>(n)], nodes[3 * static_cast<size_t>(n) + 1], nodes[3 * static_cast<size_t>(n) + 2] } };
    }
    int32_t vertex(const Val& v) {
        const auto it = vnode.find(v.r);

        if (it != vnode.end()) {
            return it->second;
        }

        const auto* a = Model::part(M.ent(v), "VERTEX_POINT");

        if (!a) {
            throw std::runtime_error(M.path + ": expected a VERTEX_POINT");
        }

        const int32_t n = add(M.point((*a)[1]));
        vnode[v.r] = n;
        return n;
    }

    // the nodes of EDGE_CURVE(name, start, end, curve, same_sense), start to end
    const std::vector<int32_t>& edge(long id) {
        const auto it = edges.find(id);

        if (it != edges.end()) {
            return it->second;
        }

        const auto* a = Model::part(M.ent(id), "EDGE_CURVE");

        if (!a || a->size() < 5) {
            throw std::runtime_error(M.path + ": expected an EDGE_CURVE");
        }

        const int32_t vs = vertex((*a)[1]), ve = vertex((*a)[2]);
        const bool sense = Model::flag((*a)[4]);
        std::shared_ptr<Curve> c = M.curve((*a)[3].r);
        std::vector<int32_t> ids{ vs };

        if (c->k != Curve::NONE && c->k != Curve::LINE) {
            const A3 ps = at(vs), pe = at(ve);
            double ts = c->param(ps), te = c->param(pe);

            if (c->periodic()) {
                if (sense) {
                    while (te <= ts + 1e-12) {
                        te += 2 * kPi;
                    }

                    if (vs != ve && te - ts > 2 * kPi) {
                        te -= 2 * kPi;
                    }
                } else {
                    while (te >= ts - 1e-12) {
                        te -= 2 * kPi;
                    }

                    if (vs != ve && ts - te > 2 * kPi) {
                        te += 2 * kPi;
                    }
                }
            } else if (vs == ve || norm(ps - pe) <= 1e-9 * std::max(1.0, norm(ps))) {   // a closed curve
                double a0, a1;
                c->range(a0, a1);
                ts = sense ? a0 : a1;
                te = sense ? a1 : a0;
            }

            // samples: the chord deviation and the turn below the tolerances, recursively
            std::vector<double> T;
            sample(*c, ts, te, 0, T);

            for (size_t i = 1; i + 1 < T.size(); ++i) {
                ids.push_back(add(c->eval(T[i])));
            }
        } else if (c->k == Curve::LINE && hmax > 0) {
            const A3 ps = at(vs), pe = at(ve);
            const int n = static_cast<int>(std::ceil(norm(pe - ps) / hmax));

            for (int i = 1; i < n; ++i) {
                ids.push_back(add(ps + (static_cast<double>(i) / n) * (pe - ps)));
            }
        } else if (c->k == Curve::NONE) {
            M.note("an edge on an unsupported curve: a straight segment");
        }

        ids.push_back(ve);

        if (vs == ve && ids.size() == 2) {   // a closed edge, not sampled
            ids.pop_back();
        }

        return edges[id] = ids;
    }
    void sample(const Curve& c, double a, double b, int depth, std::vector<double>& T) {
        // an initial split (a circle's arc: by the angle; a B-spline: by its spans)
        int n0 = 1;

        if (depth == 0) {
            if (c.periodic()) {
                const double rr = std::max(c.r, c.r2);
                double step = amax;

                if (rr > tol) {
                    step = std::min(step, 2 * std::acos(1 - tol / rr));
                }

                if (hmax > 0 && rr > 0) {
                    step = std::min(step, hmax / rr);
                }

                n0 = std::max(1, static_cast<int>(std::ceil(std::fabs(b - a) / step)));
                T.clear();

                for (int i = 0; i <= n0; ++i) {
                    T.push_back(a + (b - a) * i / n0);
                }

                return;
            }

            n0 = c.k == Curve::BSPLINE ? std::max(2, 2 * static_cast<int>(c.cp.size())) :
                 c.k == Curve::POLY ? static_cast<int>(c.cp.size() - 1) : 1;
            T.clear();
            T.push_back(a);

            for (int i = 0; i < n0; ++i) {
                sample(c, a + (b - a) * i / n0, a + (b - a) * (i + 1) / n0, 1, T);
            }

            return;
        }

        const A3 pa = c.eval(a), pb = c.eval(b), pm = c.eval(0.5 * (a + b));
        const A3 ab = pb - pa;
        const double L = norm(ab);
        const double dev = L > 0 ? norm(cross(pm - pa, ab)) / L : norm(pm - pa);
        // the turn: the angle between the two halves
        const A3 h1 = pm - pa, h2 = pb - pm;
        const double c12 = norm(h1) > 0 && norm(h2) > 0 ? dot(h1, h2) / (norm(h1) * norm(h2)) : 1;
        const bool split = depth < 14 && (dev > tol || c12 < std::cos(amax) || (hmax > 0 && L > hmax));

        if (split && c.k != Curve::POLY) {
            sample(c, a, 0.5 * (a + b), depth + 1, T);
            sample(c, 0.5 * (a + b), b, depth + 1, T);
        } else {
            T.push_back(b);
        }
    }

    // one face: its loops (each a closed list of nodes), its surface, its sense
    // flip: the other reading of a face on a closed surface bounded by closed loops
    // alone (the loops' inside or their complement, with the normal the other way);
    // ambiguous: set when the face is such a one. Its triangles into `out`.
    bool face(const Surf& S, const std::vector<std::vector<int32_t>>& loops, const std::vector<int32_t>& poles, bool sense,
              bool flip, bool& ambiguous, std::vector<int32_t>& out, std::string& why) {
        ambiguous = false;

        if (S.k == Surf::NONE) {
            why = "an unsupported surface";
            return false;
        }

        const double Pu = S.period_u(), Pv = S.period_v();

        // 1. each loop in (u, v): inverted, unwrapped along it; a singular point
        //    (a pole, an apex) split in two, at its neighbours' u
        struct UV {
            double u, v;
            int32_t id;
            bool sing;
        };
        std::vector<std::vector<UV>> L;
        std::vector<double> wu, wv;   // each loop's winding (a multiple of the period, or 0)

        for (const auto& lp : loops) {
            std::vector<UV> q;
            double hu = 0, hv = 0;
            bool hint = false;

            for (const int32_t n : lp) {   // inverted (the last point a hint for the next)
                UV x;
                bool sing;
                S.inv(at(n), x.u, x.v, sing, hu, hv, hint);
                x.id = n;
                x.sing = sing;

                if (!sing) {
                    hu = x.u;
                    hv = x.v;
                    hint = true;
                }

                q.push_back(x);
            }

            // unwrapped along the loop, never across a singular point (u is free there):
            // the loop starts just after its first one
            size_t s0 = 0;

            for (size_t i = 0; i < q.size(); ++i)
                if (q[i].sing) {
                    s0 = (i + 1) % q.size();
                    break;
                }

            std::rotate(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(s0), q.end());
            bool have = false;

            for (UV& x : q) {
                if (x.sing) {
                    have = false;   // (a new stretch after it: its own unwrapping)
                    continue;
                }

                if (have && Pu > 0) {
                    x.u = hu + wrap(x.u - hu, Pu);
                }

                if (have && Pv > 0) {
                    x.v = hv + wrap(x.v - hv, Pv);
                }

                hu = x.u;
                hv = x.v;
                have = true;
            }

            // the loop's winding: the unwrapped step from its last point back to its first
            double U0 = 0, V0 = 0, Ue = 0, Ve = 0;
            bool any = false;

            for (const UV& x : q)
                if (!x.sing) {
                    if (!any) {
                        U0 = x.u;
                        V0 = x.v;
                    }

                    Ue = x.u;
                    Ve = x.v;
                    any = true;
                }

            double w1 = 0, w2 = 0;
            bool through = false;   // through a singular point: u is free there, so no winding

            for (const UV& x : q) {
                through = through || x.sing;
            }

            if (any && !through) {
                if (Pu > 0) {
                    w1 = (Ue + wrap(U0 - Ue, Pu)) - U0;
                }

                if (Pv > 0) {
                    w2 = (Ve + wrap(V0 - Ve, Pv)) - V0;
                }
            }

            // the singular points: at the u of the points before and after
            std::vector<UV> r;
            const size_t m = q.size();

            for (size_t i = 0; i < m; ++i) {
                if (!q[i].sing) {
                    r.push_back(q[i]);
                    continue;
                }

                const UV& pv = q[(i + m - 1) % m];
                const UV& nx = q[(i + 1) % m];
                UV a = q[i], b = q[i];
                a.u = pv.sing ? q[i].u : pv.u;
                b.u = nx.sing ? q[i].u : nx.u;

                r.push_back(a);

                if (std::fabs(b.u - a.u) > 1e-12) {
                    r.push_back(b);
                }
            }

            L.push_back(r);
            wu.push_back(std::fabs(w1) > 0.5 * Pu && Pu > 0 ? w1 : 0);
            wv.push_back(std::fabs(w2) > 0.5 * Pv && Pv > 0 ? w2 : 0);

        }

        // 2. the domain's polygons: closed loops as they are; loops round the surface
        //    cut open along a seam and joined (a band: two of them; a cap: one and a pole)
        std::vector<std::vector<UV>> polys;
        std::vector<int> wrapping;

        for (size_t i = 0; i < L.size(); ++i) {
            if (wu[i] != 0 || wv[i] != 0) {
                wrapping.push_back(static_cast<int>(i));
            } else if (L[i].size() >= 3) {
                polys.push_back(L[i]);
            }
        }

        const int32_t none = -1;
        auto seam = [&](const UV & a, const UV & b, std::vector<UV>& dst, std::vector<int32_t>& made) {
            // points from a to b (exclusive), new nodes on the surface; `made`: their ids
            // (a straight uv line, bent in 3-D: its length by a few samples)
            double L3 = 0;
            A3 pr = S.eval(a.u, a.v);

            for (int k = 1; k <= 8; ++k) {
                const A3 pk = S.eval(a.u + (b.u - a.u) * k / 8, a.v + (b.v - a.v) * k / 8);
                L3 += norm(pk - pr);
                pr = pk;
            }

            int n = std::max(1, static_cast<int>(std::ceil(L3 / std::max(1e-12, hmax > 0 ? std::min(hmax, 8 * tol) : 8 * tol))));
            n = std::min(n, 256);

            if (made.empty()) {
                for (int k = 1; k < n; ++k) {
                    made.push_back(add(S.eval(a.u + (b.u - a.u) * k / n, a.v + (b.v - a.v) * k / n)));
                }
            }

            const int nm = static_cast<int>(made.size()) + 1;

            for (int k = 1; k < nm; ++k) {
                dst.push_back({ a.u + (b.u - a.u) * k / nm, a.v + (b.v - a.v) * k / nm, made[static_cast<size_t>(k - 1)], false });
            }
        };
        // rotate loop i to start at its point nearest (in u, mod the period) to uc, shifted near uc
        auto open_at = [&](int i, double uc, double vc, bool by_v) {
            std::vector<UV> q = L[static_cast<size_t>(i)];
            size_t best = 0;
            double bd = 1e300;

            for (size_t j = 0; j < q.size(); ++j) {
                if (q[j].sing) {
                    continue;
                }

                const double d = by_v ? std::fabs(wrap(q[j].v - vc, Pv)) : std::fabs(wrap(q[j].u - uc, Pu));

                if (d < bd) {
                    bd = d;
                    best = j;
                }
            }

            const double W = by_v ? wv[static_cast<size_t>(i)] : wu[static_cast<size_t>(i)];
            std::vector<UV> r;

            for (size_t j = 0; j < q.size(); ++j) {
                UV x = q[(best + j) % q.size()];

                if (best + j >= q.size()) {
                    (by_v ? x.v : x.u) += W;
                }

                r.push_back(x);
            }

            const double sh = by_v ? vc + wrap(r[0].v - vc, Pv) - r[0].v : uc + wrap(r[0].u - uc, Pu) - r[0].u;

            for (UV& x : r) {
                (by_v ? x.v : x.u) += sh;
            }

            UV end = r[0];
            (by_v ? end.v : end.u) += W;
            r.push_back(end);   // (the start again, one period on)
            return r;
        };
        double win_u = 0;   // the domain's window in u (for the closed loops' shift)
        bool have_win = false;

        if (wrapping.size() == 2) {   // a band between two loops round it
            const int A = wrapping[0], B = wrapping[1];
            const bool byv = wu[static_cast<size_t>(A)] == 0;   // (round in v: a torus' tube)

            if ((byv && wu[static_cast<size_t>(B)] != 0) || (!byv && wu[static_cast<size_t>(B)] == 0)) {
                why = "two loops round the surface in different directions";
                return false;
            }

            const double uc = L[static_cast<size_t>(A)][0].u, vc = L[static_cast<size_t>(A)][0].v;
            std::vector<UV> a = open_at(A, uc, vc, byv);
            std::vector<UV> b = open_at(B, byv ? uc : a[0].u, byv ? a[0].v : vc, byv);
            const double WA = byv ? wv[static_cast<size_t>(A)] : wu[static_cast<size_t>(A)];
            const double WB = byv ? wv[static_cast<size_t>(B)] : wu[static_cast<size_t>(B)];

            if ((WA > 0) == (WB > 0)) {   // (the same way round: B reversed)
                std::reverse(b.begin(), b.end());
            }

            // B must run from A's far end back: its start near a.back()
            const double bshift = byv ? (a.back().v - b[0].v) : (a.back().u - b[0].u);
            const double per = byv ? Pv : Pu;
            const double k = std::round(bshift / per);

            for (UV& x : b) {
                (byv ? x.v : x.u) += k * per;
            }

            std::vector<UV> poly = a;
            std::vector<int32_t> seam1, seam2;
            seam(a.back(), b.front(), poly, seam1);
            poly.insert(poly.end(), b.begin(), b.end());
            // the other seam: the same nodes, one period back
            UV bs = b.back(), as = a.front();
            std::vector<UV> back;
            seam(as, bs, back, seam1);   // (as -> bs, reversed below; seam1 reused: same nodes)
            std::reverse(back.begin(), back.end());
            poly.insert(poly.end(), back.begin(), back.end());
            // (the last point of b and the first of a: the seam's ends; closed)
            polys.push_back(poly);
            win_u = byv ? 0 : a.front().u + 0.5 * WA;
            have_win = !byv;
        } else if (wrapping.size() == 1) {   // a cap: one loop round it, closed at a pole / apex
            const int A = wrapping[0];

            if (wu[static_cast<size_t>(A)] == 0) {
                why = "a loop round the surface's v direction alone";
                return false;
            }

            double vp;

            if (S.k == Surf::SPHERE) {
                // the pole on the face's side: left of a loop running +u, when the face is
                // on the surface's normal side
                const bool up = (wu[static_cast<size_t>(A)] > 0) == sense;
                vp = up ? 0.5 * kPi : -0.5 * kPi;
            } else if (!S.singular_v(vp)) {
                // a revolved curve's end on the axis
                if (S.k == Surf::REVOL) {
                    const bool up = (wu[static_cast<size_t>(A)] > 0) == sense;
                    vp = up ? S.v1 : S.v0;
                } else {
                    why = "a single loop round a surface with no pole";
                    return false;
                }
            }

            std::vector<UV> a = open_at(A, L[static_cast<size_t>(A)][0].u, 0, false);
            int32_t pole = poles.empty() ? none : poles[0];

            if (pole == none) {
                pole = add(S.eval(a[0].u, vp));
            }

            std::vector<UV> poly = a;
            std::vector<int32_t> s1;
            UV p1{ a.back().u, vp, pole, true }, p0{ a.front().u, vp, pole, true };
            seam(a.back(), p1, poly, s1);
            // along the pole: several points, all the pole's node
            const int np = std::max(2, static_cast<int>(std::ceil(std::fabs(a.back().u - a.front().u) / amax)));

            for (int k = 0; k <= np; ++k) {
                poly.push_back({ a.back().u + (a.front().u - a.back().u) * k / np, vp, pole, true });
            }

            std::vector<UV> back;
            seam(a.front(), p0, back, s1);
            std::reverse(back.begin(), back.end());
            poly.insert(poly.end(), back.begin(), back.end());
            polys.push_back(poly);
            win_u = a.front().u + 0.5 * wu[static_cast<size_t>(A)];
            have_win = true;
        } else if (wrapping.size() > 2) {
            why = std::to_string(wrapping.size()) + " loops round the surface";
            return false;
        }

        // an outer boundary runs counter-clockwise round the face (in its sense), a
        // hole clockwise
        bool any_outer = false;

        for (const auto& p : polys) {
            double a2 = 0;

            for (size_t k = 0, j = p.size() - 1; k < p.size(); j = k++) {
                a2 += p[j].u * p[k].v - p[k].u * p[j].v;
            }

            any_outer = any_outer || (sense ? a2 : -a2) > 0;
        }

        const bool closed_surf = S.k == Surf::SPHERE || (S.per_u && S.per_v) || (S.per_u && S.k == Surf::REVOL);

        if (wrapping.empty() && closed_surf && !polys.empty()) {
            // the loops' inside, or all but it: the file's orientation decides (some
            // exporters disagree; the caller tries the other reading when this one
            // crosses the rest of the shell)
            ambiguous = true;

            if (flip) {
                any_outer = !any_outer;
            }
        }

        if (wrapping.empty() && !any_outer && !closed_surf && !polys.empty()) {
            // holes only, on an open surface: loops oriented the other way (some
            // exporters): the widest one is the boundary (even-odd does the rest)
            any_outer = true;
        }

        if (wrapping.empty() && !any_outer) {
            // no outer loop (none at all, or holes only): the whole closed surface (a
            // sphere, a torus, ..) less the holes -- its parameter rectangle, the sides a
            // seam (both copies the same nodes) or a pole (all one node), the seam where
            // no hole is
            auto gap = [&](bool in_u, double P0) {   // a u (v) that no loop covers, mod the period
                if (P0 <= 0 || polys.empty()) {
                    return 0.0;
                }

                double best = 0, bd = -1;

                for (int k = 0; k < 720; ++k) {
                    const double c = P0 * k / 720;
                    double dmin = 1e300;

                    for (const auto& p : polys) {
                        double lo = 1e300, hi = -1e300;

                        for (const UV& x : p) {
                            lo = std::min(lo, in_u ? x.u : x.v);
                            hi = std::max(hi, in_u ? x.u : x.v);
                        }

                        // c inside [lo, hi] (mod P0)?
                        const double cc = lo + std::fmod(std::fmod(c - lo, P0) + P0, P0);

                        if (cc <= hi) {
                            dmin = -1;
                            break;
                        }

                        dmin = std::min(dmin, std::min(cc - hi, lo + P0 - cc));
                    }

                    if (dmin > bd) {
                        bd = dmin;
                        best = c;
                    }
                }

                return best;
            };
            double ua = 0, ub = 0, va = 0, vb = 0;
            bool pole_a = false, pole_b = false;

            if (S.k == Surf::SPHERE) {
                ua = gap(true, Pu);
                ub = ua + 2 * kPi;
                va = -0.5 * kPi;
                vb = 0.5 * kPi;
                pole_a = pole_b = true;
            } else if (S.per_u && S.per_v) {
                ua = (S.k == Surf::BSPLINE ? S.u0 : 0) + gap(true, Pu);
                ub = ua + Pu;
                va = (S.k == Surf::BSPLINE ? S.v0 : 0) + gap(false, Pv);
                vb = va + Pv;
            } else if (S.per_u && S.k == Surf::REVOL) {
                ua = gap(true, Pu);
                ub = ua + 2 * kPi;
                va = S.v0;
                vb = S.v1;
                pole_a = pole_b = true;
            } else {
                why = polys.empty() ? "no loop, and not a closed surface" : "only holes, and not a closed surface";
                return false;
            }

            // the holes into the rectangle's window (by whole periods)
            for (auto& p : polys) {
                double ulo = 1e300, vlo2 = 1e300;

                for (const UV& x : p) {
                    ulo = std::min(ulo, x.u);
                    vlo2 = std::min(vlo2, x.v);
                }

                const double shu = Pu > 0 ? Pu * std::floor((ulo - ua) / Pu) : 0;
                const double shv = Pv > 0 ? Pv * std::floor((vlo2 - va) / Pv) : 0;

                for (UV& x : p) {
                    x.u -= shu;
                    x.v -= shv;
                }
            }

            auto pole_node = [&](double v) {   // a node there already (a VERTEX_LOOP's, a loop's
                const A3 p = S.eval(ua, v);    // through the pole), else a new one

                for (const int32_t n : poles)
                    if (norm(at(n) - p) <= 10 * tol) {
                        return n;
                    }

                for (const auto& lp : L)
                    for (const UV& x : lp)
                        if (x.sing && norm(at(x.id) - p) <= 10 * tol) {
                            return x.id;
                        }

                return add(p);
            };
            // the sides' divisions: the angle, the chord tolerance (at the largest
            // radius the side turns about) and the size
            double ru = 0, rv = 0;

            for (int k = 0; k <= 8; ++k) {
                const double uu = ua + (ub - ua) * k / 8, vv = va + (vb - va) * k / 8, e = 1e-6;
                ru = std::max(ru, norm(S.eval(ua + e, vv) - S.eval(ua - e, vv)) / (2 * e));
                rv = std::max(rv, norm(S.eval(uu, va + 0.5 * (vb - va) + e) - S.eval(uu, va + 0.5 * (vb - va) - e)) / (2 * e));
            }

            auto divs = [&](double span, double rad, int lo) {
                double step = amax;

                if (rad > tol) {
                    step = std::min(step, 2 * std::acos(1 - tol / rad));
                }

                int n = static_cast<int>(std::ceil(span / step));

                if (hmax > 0) {
                    n = std::max(n, static_cast<int>(std::ceil(span * rad / hmax)));
                }

                return std::min(4096, std::max(lo, n));
            };
            const int nU = divs(ub - ua, ru, 3), nV = divs(vb - va, rv, 2);
            std::vector<int32_t> bot(static_cast<size_t>(nU) + 1), top(static_cast<size_t>(nU) + 1),
                left(static_cast<size_t>(nV) + 1), right(static_cast<size_t>(nV) + 1);
            const int32_t pa = pole_a ? pole_node(va) : none, pb = pole_b ? pole_node(vb) : none;

            for (int i = 0; i <= nU; ++i) {
                const double u = ua + (ub - ua) * i / nU;
                bot[static_cast<size_t>(i)] = pole_a ? pa : (i == nU && S.per_u ? bot[0] : add(S.eval(u, va)));
            }

            for (int i = 0; i <= nU; ++i) {
                const double u = ua + (ub - ua) * i / nU;
                top[static_cast<size_t>(i)] = pole_b ? pb : S.per_v ? bot[static_cast<size_t>(i)] :
                                              (i == nU && S.per_u ? top[0] : add(S.eval(u, vb)));
            }

            left[0] = bot[0];
            left[static_cast<size_t>(nV)] = top[0];
            right[0] = bot[static_cast<size_t>(nU)];
            right[static_cast<size_t>(nV)] = top[static_cast<size_t>(nU)];

            for (int j = 1; j < nV; ++j) {
                const double v = va + (vb - va) * j / nV;
                left[static_cast<size_t>(j)] = add(S.eval(ua, v));
                right[static_cast<size_t>(j)] = S.per_u ? left[static_cast<size_t>(j)] : add(S.eval(ub, v));
            }

            std::vector<UV> poly;

            for (int i = 0; i < nU; ++i) {
                poly.push_back({ ua + (ub - ua) * i / nU, va, bot[static_cast<size_t>(i)], pole_a });
            }

            for (int j = 0; j < nV; ++j) {
                poly.push_back({ ub, va + (vb - va) * j / nV, right[static_cast<size_t>(j)], false });
            }

            for (int i = nU; i > 0; --i) {
                poly.push_back({ ua + (ub - ua) * i / nU, vb, top[static_cast<size_t>(i)], pole_b });
            }

            for (int j = nV; j > 0; --j) {
                poly.push_back({ ua, va + (vb - va) * j / nV, left[static_cast<size_t>(j)], false });
            }

            polys.push_back(poly);
            win_u = 0.5 * (ua + ub);
            have_win = true;
        }

        if (polys.empty()) {
            why = "no loop with area";
            return false;
        }

        // the closed loops shifted into one window of the periodic u (the widest one's)
        if (Pu > 0) {
            if (!have_win) {
                double wd = -1;

                for (size_t i = 0; i < polys.size(); ++i) {
                    double lo = 1e300, hi = -1e300;

                    for (const UV& x : polys[i]) {
                        lo = std::min(lo, x.u);
                        hi = std::max(hi, x.u);
                    }

                    if (hi - lo > wd) {
                        wd = hi - lo;
                        win_u = 0.5 * (lo + hi);
                    }
                }
            }

            for (auto& p : polys) {
                double lo = 1e300, hi = -1e300;

                for (const UV& x : p) {
                    lo = std::min(lo, x.u);
                    hi = std::max(hi, x.u);
                }

                const double sh = Pu * std::round((win_u - 0.5 * (lo + hi)) / Pu);

                if (sh != 0 && hi - lo < 0.999 * Pu) {
                    for (UV& x : p) {
                        x.u += sh;
                    }
                }
            }
        }

        // 3. the plane's metric: u and v scaled to mm (at the domain's middle)
        double ulo = 1e300, uhi = -1e300, vlo = 1e300, vhi = -1e300;

        for (const auto& p : polys)
            for (const UV& x : p) {
                ulo = std::min(ulo, x.u);
                uhi = std::max(uhi, x.u);
                vlo = std::min(vlo, x.v);
                vhi = std::max(vhi, x.v);
            }

        const double um = 0.5 * (ulo + uhi), vm = 0.5 * (vlo + vhi);
        const double eu = std::max(1e-9, 1e-4 * (uhi - ulo)), ev = std::max(1e-9, 1e-4 * (vhi - vlo));
        double su = norm(S.eval(um + eu, vm) - S.eval(um - eu, vm)) / (2 * eu);
        double sv = norm(S.eval(um, vm + ev) - S.eval(um, vm - ev)) / (2 * ev);

        if (!(su > 0)) {
            su = 1;
        }

        if (!(sv > 0)) {
            sv = 1;
        }

        // 4. the points: the polygons', then interior ones on a curved surface
        std::vector<double> P;
        std::vector<int32_t> pid;       // node (or -1: an interior point, made below)
        std::vector<int> segs;
        std::map<std::pair<long long, long long>, int> key;
        auto qk = [&](double u, double v) {
            return std::make_pair(static_cast<long long>(std::llround(u * su / (1e-7 * std::max(tol, 1e-12)))),
                                  static_cast<long long>(std::llround(v * sv / (1e-7 * std::max(tol, 1e-12)))));
        };
        auto point = [&](double u, double v, int32_t id) {
            const auto k = qk(u, v);
            const auto it = key.find(k);

            if (it != key.end()) {
                return it->second;
            }

            const int i = static_cast<int>(pid.size());
            key[k] = i;
            P.push_back(u * su);
            P.push_back(v * sv);
            pid.push_back(id);
            return i;
        };
        std::vector<std::vector<int>> pl;   // the polygons as point indices

        for (const auto& p : polys) {
            std::vector<int> ix;

            for (const UV& x : p) {
                const int i = point(x.u, x.v, x.id);

                if (ix.empty() || ix.back() != i) {
                    ix.push_back(i);
                }
            }

            while (ix.size() > 1 && ix.front() == ix.back()) {
                ix.pop_back();
            }

            if (ix.size() < 3) {
                continue;
            }

            for (size_t k = 0; k < ix.size(); ++k) {
                segs.push_back(ix[k]);
                segs.push_back(ix[(k + 1) % ix.size()]);
            }

            pl.push_back(ix);
        }

        auto inside = [&](double x, double y) {   // even-odd over the polygons (scaled)
            bool in = false;

            for (const auto& ix : pl)
                for (size_t k = 0, j = ix.size() - 1; k < ix.size(); j = k++) {
                    const double xi = P[2 * static_cast<size_t>(ix[k])], yi = P[2 * static_cast<size_t>(ix[k]) + 1];
                    const double xj = P[2 * static_cast<size_t>(ix[j])], yj = P[2 * static_cast<size_t>(ix[j]) + 1];

                    if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                        in = !in;
                    }
                }

            return in;
        };

        if (S.k != Surf::PLANE || hmax > 0) {   // interior points: a grid fine enough for the tolerance (and --size)
            int nu = 4, nv = 4;

            for (int pass = 0; pass < 8; ++pass) {
                double devu = 0, devv = 0;

                for (int i = 0; i <= nu; ++i)
                    for (int j = 0; j <= nv; ++j) {
                        const double u = ulo + (uhi - ulo) * i / nu, v = vlo + (vhi - vlo) * j / nv;
                        const double du = (uhi - ulo) / nu, dv = (vhi - vlo) / nv;

                        if (i < nu) {
                            const A3 a = S.eval(u, v), b = S.eval(u + du, v), m = S.eval(u + 0.5 * du, v);
                            devu = std::max(devu, norm(m - 0.5 * (a + b)));
                        }

                        if (j < nv) {
                            const A3 a = S.eval(u, v), b = S.eval(u, v + dv), m = S.eval(u, v + 0.5 * dv);
                            devv = std::max(devv, norm(m - 0.5 * (a + b)));
                        }
                    }

                bool more = false;

                if (devu > tol && nu < 512) {
                    nu *= 2;
                    more = true;
                }

                if (devv > tol && nv < 512) {
                    nv *= 2;
                    more = true;
                }

                if (!more) {
                    break;
                }
            }

            if (hmax > 0) {
                nu = std::max(nu, static_cast<int>(std::ceil((uhi - ulo) * su / hmax)));
                nv = std::max(nv, static_cast<int>(std::ceil((vhi - vlo) * sv / hmax)));
            }

            // (at most 40000 a face: coarser, evenly, beyond that)
            while (static_cast<double>(nu) * nv > 40000.0) {
                nu = std::max(2, nu * 3 / 4);
                nv = std::max(2, nv * 3 / 4);
            }

            const double gx = (uhi - ulo) * su / nu, gy = (vhi - vlo) * sv / nv;
            const double clear = 0.5 * std::min(gx, gy);
            // the boundary segments in buckets of the grid's cell
            const double bc = std::max(gx, gy);
            std::unordered_map<long long, std::vector<size_t>> bk;
            auto bkey = [&](long long i, long long j) {
                return i * 1000003LL + j;
            };

            for (size_t k = 0; k + 1 < segs.size(); k += 2) {
                const double ax = P[2 * static_cast<size_t>(segs[k])], ay = P[2 * static_cast<size_t>(segs[k]) + 1];
                const double bx = P[2 * static_cast<size_t>(segs[k + 1])], by = P[2 * static_cast<size_t>(segs[k + 1]) + 1];
                const long long i0 = static_cast<long long>(std::floor((std::min(ax, bx) - clear) / bc));
                const long long i1 = static_cast<long long>(std::floor((std::max(ax, bx) + clear) / bc));
                const long long j0 = static_cast<long long>(std::floor((std::min(ay, by) - clear) / bc));
                const long long j1 = static_cast<long long>(std::floor((std::max(ay, by) + clear) / bc));

                if ((i1 - i0 + 1) * (j1 - j0 + 1) > 1000000) {
                    continue;   // (a degenerate span)
                }

                for (long long i = i0; i <= i1; ++i)
                    for (long long j = j0; j <= j1; ++j) {
                        bk[bkey(i, j)].push_back(k);
                    }
            }

            for (int i = 1; i < nu; ++i)
                for (int j = 1; j < nv; ++j) {
                    const double u = ulo + (uhi - ulo) * i / nu, v = vlo + (vhi - vlo) * j / nv;
                    const double x = u * su, y = v * sv;

                    if (!inside(x, y)) {
                        continue;
                    }

                    bool near = false;   // (clear of the boundary)
                    const auto itb = bk.find(bkey(static_cast<long long>(std::floor(x / bc)), static_cast<long long>(std::floor(y / bc))));

                    if (itb == bk.end()) {
                        point(u, v, none);
                        continue;
                    }

                    for (const size_t k : itb->second) {
                        if (near) {
                            break;
                        }

                        const double ax = P[2 * static_cast<size_t>(segs[k])], ay = P[2 * static_cast<size_t>(segs[k]) + 1];
                        const double bx = P[2 * static_cast<size_t>(segs[k + 1])], by = P[2 * static_cast<size_t>(segs[k + 1]) + 1];
                        const double ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
                        const double t = l2 > 0 ? std::max(0.0, std::min(1.0, ((x - ax) * ex + (y - ay) * ey) / l2)) : 0.0;
                        const double dx = x - (ax + t * ex), dy = y - (ay + t * ey);
                        near = dx * dx + dy * dy < clear * clear;
                    }

                    if (!near) {
                        point(u, v, none);
                    }
                }
        }

        // 5. the constrained triangulation, the triangles inside, mapped back
        std::vector<int> T, comp;
        std::vector<char> hull;

        try {
            cdt2d(P, segs, T, comp, hull);
        } catch (const std::exception& e) {
            why = e.what();
            return false;
        }

        std::vector<int> keep;   // per component: -1 unknown, 0 out, 1 in

        for (size_t t = 0; t < comp.size(); ++t) {
            const int c = comp[t];

            if (c >= static_cast<int>(keep.size())) {
                keep.resize(static_cast<size_t>(c) + 1, -1);
            }

            if (keep[static_cast<size_t>(c)] < 0) {
                const int* v = &T[3 * t];
                const double x = (P[2 * static_cast<size_t>(v[0])] + P[2 * static_cast<size_t>(v[1])] + P[2 * static_cast<size_t>(v[2])]) / 3;
                const double y = (P[2 * static_cast<size_t>(v[0]) + 1] + P[2 * static_cast<size_t>(v[1]) + 1] +
                                  P[2 * static_cast<size_t>(v[2]) + 1]) / 3;
                keep[static_cast<size_t>(c)] = inside(x, y) ? 1 : 0;
            }
        }

        for (size_t i = 0; i < pid.size(); ++i)
            if (pid[i] < 0) {
                pid[i] = add(S.eval(P[2 * i] / su, P[2 * i + 1] / sv));
            }

        size_t made = 0;

        for (size_t t = 0; t < comp.size(); ++t) {
            if (keep[static_cast<size_t>(comp[t])] != 1) {
                continue;
            }

            int32_t a = pid[static_cast<size_t>(T[3 * t])], b = pid[static_cast<size_t>(T[3 * t + 1])],
                    c = pid[static_cast<size_t>(T[3 * t + 2])];

            if (a == b || b == c || a == c) {
                continue;   // (at a pole)
            }

            // the face's normal: the surface's, flipped when not in its sense
            const double cu = (P[2 * static_cast<size_t>(T[3 * t])] + P[2 * static_cast<size_t>(T[3 * t + 1])] +
                               P[2 * static_cast<size_t>(T[3 * t + 2])]) / (3 * su);
            const double cv = (P[2 * static_cast<size_t>(T[3 * t]) + 1] + P[2 * static_cast<size_t>(T[3 * t + 1]) + 1] +
                               P[2 * static_cast<size_t>(T[3 * t + 2]) + 1]) / (3 * sv);
            const A3 n = (sense != (flip && ambiguous) ? 1.0 : -1.0) * S.normal(cu, cv);

            if (dot(cross(at(b) - at(a), at(c) - at(a)), n) < 0) {
                std::swap(b, c);
            }

            out.insert(out.end(), { a, b, c });
            ++made;
        }

        if (made == 0) {
            why = "no triangles inside its loops";
            return false;
        }

        return true;
    }
};

}  // namespace

Mesh read_step(const std::string& path, const StepOptions& o, StepStats* st) {
    StepStats S;
    Model M;
    M.path = path;
    parse_step(path, M.E);
    S.entities = M.E.size();
    M.units();
    S.unit = M.len;

    // the solids: (label, outer shell, void shells); else the shells of a surface model
    struct Solid {
        std::vector<long> shells;
    };
    std::vector<Solid> solids;
    std::vector<long> ids;

    for (const auto& kv : M.E) {
        ids.push_back(kv.first);
    }

    std::sort(ids.begin(), ids.end());

    for (const long id : ids) {
        const Ent& e = M.ent(id);
        const std::vector<Val>* a;

        if ((a = Model::part(e, "BREP_WITH_VOIDS"))) {   // (name, outer, (ORIENTED_CLOSED_SHELL ..))
            Solid s;
            s.shells.push_back((*a)[1].r);

            for (const Val& v : Model::list((*a)[2])) {
                const auto* oc = Model::part(M.ent(v), "ORIENTED_CLOSED_SHELL");   // (name, *, shell, orientation)
                s.shells.push_back(oc ? (*oc)[2].r : v.r);
            }

            solids.push_back(s);
        } else if ((a = Model::part(e, "MANIFOLD_SOLID_BREP"))) {
            solids.push_back({ { (*a)[1].r } });
        }
    }

    if (solids.empty())
        for (const long id : ids) {
            const auto* a = Model::part(M.ent(id), "SHELL_BASED_SURFACE_MODEL");

            if (a)
                for (const Val& v : Model::list((*a)[1])) {
                    solids.push_back({ { v.r } });
                }
        }

    if (solids.empty()) {
        throw std::runtime_error(path + ": no B-rep solid or shell (MANIFOLD_SOLID_BREP, BREP_WITH_VOIDS, "
                                 "SHELL_BASED_SURFACE_MODEL)");
    }

    // the model's size (its vertices): the default tolerance
    Builder B(M, o);
    {
        A3 lo{ { 1e300, 1e300, 1e300 } }, hi{ { -1e300, -1e300, -1e300 } };
        A3 vlo{ { 1e300, 1e300, 1e300 } }, vhi{ { -1e300, -1e300, -1e300 } };   // (the vertices')
        auto grow = [&](const A3 & p, double r) {
            for (size_t k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], p[k] - r);
                hi[k] = std::max(hi[k], p[k] + r);
            }
        };

        for (const long id : ids) {   // the points, and the round primitives' reach
            const Ent& e = M.ent(id);
            const std::vector<Val>* a;

            auto placed3 = [&](const Val & v) {  // (a 2-D placement: a pcurve's, in a parameter plane)
                return v.k == Val::REF && Model::part(M.ent(v), "AXIS2_PLACEMENT_3D") != nullptr;
            };

            if ((a = Model::part(e, "VERTEX_POINT"))) {
                const A3 p = M.point((*a)[1]);

                for (size_t k = 0; k < 3; ++k) {
                    vlo[k] = std::min(vlo[k], p[k]);
                    vhi[k] = std::max(vhi[k], p[k]);
                }
            } else if (e.type == "CARTESIAN_POINT") {
                if (e.a.size() > 1 && Model::list(e.a[1]).size() == 3) {   // (3-D points only)
                    Val rv;
                    rv.k = Val::REF;
                    rv.r = id;
                    grow(M.point(rv), 0);
                }
            } else if (((a = Model::part(e, "CIRCLE")) || (a = Model::part(e, "SPHERICAL_SURFACE")) ||
                        (a = Model::part(e, "CYLINDRICAL_SURFACE"))) && placed3((*a)[1])) {
                grow(M.placement((*a)[1]).o, M.len * Model::num((*a)[2]));
            } else if ((a = Model::part(e, "TOROIDAL_SURFACE")) && placed3((*a)[1])) {
                grow(M.placement((*a)[1]).o, M.len * (Model::num((*a)[2]) + Model::num((*a)[3])));
            }
        }

        // the size: the vertices' span, unless they span next to nothing (a sphere's
        // pole): then every point's and the round primitives' (a placement far out)
        const double dall = lo[0] <= hi[0] ? norm(hi - lo) : 1.0;
        const double dv = vlo[0] <= vhi[0] ? norm(vhi - vlo) : 0.0;
        const double diag = dv >= 0.05 * dall ? dv : dall;
        B.tol = o.tol > 0 ? o.tol : 5e-4 * std::max(diag, 1e-9);
        B.amax = std::max(1.0, std::min(60.0, o.angle)) * kPi / 180;
        B.hmax = o.size;
        M.diag = diag;
    }

    S.tol = B.tol;
    S.solids = solids.size();

    for (size_t si = 0; si < solids.size(); ++si) {
        const int label = static_cast<int>(si) + 1;

        for (const long sh : solids[si].shells) {
            const Ent& se = M.ent(sh);
            const std::vector<Val>* a = Model::part(se, "CLOSED_SHELL");

            if (!a) {
                a = Model::part(se, "OPEN_SHELL");
            }

            if (!a) {
                M.note("a shell " + se.type + " not read");
                continue;
            }

            ++S.shells;
            // the shell's faces, tessellated one by one (kept apart: an ambiguous one
            // may be redone)
            struct Job {
                std::shared_ptr<Surf> surf;
                std::vector<std::vector<int32_t>> loops;
                std::vector<int32_t> poles;
                bool sense = true, amb = false, flipped = false;
                long id = 0;
                std::vector<int32_t> tri;
            };
            std::vector<Job> jobs;

            for (const Val& fv : Model::list((*a)[1])) {
                const Ent& fe = M.ent(fv);
                const std::vector<Val>* f = Model::part(fe, "ADVANCED_FACE");

                if (!f) {
                    f = Model::part(fe, "FACE_SURFACE");
                }

                ++S.faces;

                if (!f || f->size() < 4) {
                    ++S.faces_failed;
                    M.note("a face " + fe.type + " not read");
                    continue;
                }

                // (name, bounds, surface, same_sense)
                std::vector<std::vector<int32_t>> loops;
                std::vector<int32_t> poles;
                std::string why;
                bool ok = true;

                try {
                    for (const Val& bv : Model::list((*f)[1])) {
                        const Ent& be = M.ent(bv);
                        const std::vector<Val>* b = Model::part(be, "FACE_OUTER_BOUND");

                        if (!b) {
                            b = Model::part(be, "FACE_BOUND");
                        }

                        if (!b) {
                            continue;
                        }

                        const bool bor = b->size() < 3 || Model::flag((*b)[2]);
                        const Ent& le = M.ent((*b)[1]);

                        if (const auto* vl = Model::part(le, "VERTEX_LOOP")) {
                            poles.push_back(B.vertex((*vl)[1]));
                            continue;
                        }

                        const auto* el = Model::part(le, "EDGE_LOOP");

                        if (!el) {
                            M.note("a loop " + le.type + " not read");
                            continue;
                        }

                        std::vector<int32_t> lp;

                        for (const Val& oe : Model::list((*el)[1])) {
                            const auto* x = Model::part(M.ent(oe), "ORIENTED_EDGE");   // (name, *, *, edge, orientation)

                            if (!x) {
                                continue;
                            }

                            std::vector<int32_t> ids2 = B.edge((*x)[3].r);

                            if (!Model::flag((*x)[4])) {
                                std::reverse(ids2.begin(), ids2.end());
                            }

                            for (const int32_t n : ids2)
                                if (lp.empty() || lp.back() != n) {
                                    lp.push_back(n);
                                }
                        }

                        while (lp.size() > 1 && lp.front() == lp.back()) {
                            lp.pop_back();
                        }

                        if (!bor) {
                            std::reverse(lp.begin(), lp.end());
                        }

                        if (!lp.empty()) {
                            loops.push_back(lp);
                        }
                    }

                    Job j;
                    j.surf = M.surf((*f)[2].r);
                    j.loops = loops;
                    j.poles = poles;
                    j.sense = Model::flag((*f)[3]);
                    j.id = fv.r;
                    ok = B.face(*j.surf, j.loops, j.poles, j.sense, false, j.amb, j.tri, why);

                    if (ok) {
                        jobs.push_back(std::move(j));
                    }
                } catch (const std::exception& ex) {
                    ok = false;
                    why = ex.what();
                }

                if (!ok) {
                    ++S.faces_failed;
                    M.note("face #" + std::to_string(fv.r) + ": " + why);
                }
            }

            static const bool faceid = std::getenv("V2M_STEP_FACEID") != nullptr;   // (debugging)

            for (const Job& j : jobs) {
                B.tris.insert(B.tris.end(), j.tri.begin(), j.tri.end());

                for (size_t k = 0; k < j.tri.size(); k += 3) {
                    B.tlab.push_back(faceid ? static_cast<int>(j.id) : label);
                    B.tlab.push_back(0);
                }
            }
        }
    }

    Mesh m;
    m.nodes = B.nodes;
    m.tris = B.tris;
    m.tri_labels = B.tlab;
    compact_nodes(m);
    S.edges = B.edges.size();
    S.triangles = m.tris.size() / 3;
    S.nodes = m.nodes.size() / 3;
    S.notes = M.notes;

    if (st) {
        *st = S;
    }

    return m;
}

}  // namespace tn
