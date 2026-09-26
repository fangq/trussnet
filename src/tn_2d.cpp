// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_2d.cpp -- see tn_2d.h. Pixel (i, j) has its centre at (i vx, j vy) (grid
// mm); outside the image is the exterior (label 0).

#include "tn_2d.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "delaunay.h"   // the exact orient2d / incircle of the vendored predicates
#include "tn_log.h"

namespace tn {

namespace {

typedef std::chrono::steady_clock clk;
double ms_since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

enum { T_INTERIOR = 0, T_INTERFACE = 1, T_JUNCTION = 2 };

struct Moved {   // a trapped move
    bool ok;
    double x, y;
};

// ---- interface fields -------------------------------------------------------------

struct Field {
    int nx = 0, ny = 0, nlab = 0;
    double vx = 1, vy = 1;
    std::vector<float> phi;   // nlab x npix, x fastest

    float at(int l, int i, int j) const {
        if (i < 0 || j < 0 || i >= nx || j >= ny) {
            return l == 0 ? 1.0f : 0.0f;
        }

        return phi[static_cast<size_t>(l) * nx * ny + i + static_cast<size_t>(nx) * j];
    }

    // bilinear value (and gradient, mm^-1) at (x, y) mm
    double val(int l, double x, double y, double* gx = nullptr, double* gy = nullptr) const {
        const double u = x / vx, v = y / vy;
        const int i = static_cast<int>(std::floor(u)), j = static_cast<int>(std::floor(v));
        const double fx = u - i, fy = v - j;
        const double a = at(l, i, j), b = at(l, i + 1, j), c = at(l, i, j + 1), d = at(l, i + 1, j + 1);

        if (gx) {
            *gx = ((b - a) * (1 - fy) + (d - c) * fy) / vx;
            *gy = ((c - a) * (1 - fx) + (d - b) * fx) / vy;
        }

        return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
    }

    // argmax label at (x, y); the runner-up and the top-2 margin
    int label(double x, double y, int* second = nullptr, double* margin = nullptr) const {
        int best = 0, sec = -1;
        double pb = -1e30, ps = -1e30;

        for (int l = 0; l < nlab; ++l) {
            const double v = val(l, x, y);

            if (v > pb) {
                ps = pb;
                sec = best;
                pb = v;
                best = l;
            } else if (v > ps) {
                ps = v;
                sec = l;
            }
        }

        if (second) {
            *second = sec;
        }

        if (margin) {
            *margin = pb - ps;
        }

        return best;
    }

    double psi(int a, int b, double x, double y, double* gx, double* gy) const {
        double ax, ay, bx, by;
        const double pa = val(a, x, y, &ax, &ay), pb = val(b, x, y, &bx, &by);
        *gx = ax - bx;
        *gy = ay - by;
        return pa - pb;
    }

    // the largest field among the labels other than a and b
    double third(int a, int b, double x, double y) const {
        double m = -1e30;

        for (int l = 0; l < nlab; ++l)
            if (l != a && l != b) {
                m = std::max(m, val(l, x, y));
            }

        return m;
    }

    // Newton onto psi_ab = 0 from (x, y), within rad; valid if a, b stay the top two
    bool project(int a, int b, double& x, double& y, double rad) const {
        const double x0 = x, y0 = y;
        double X = x, Y = y;

        for (int it = 0; it < 12; ++it) {
            double gx, gy;
            const double p = psi(a, b, X, Y, &gx, &gy);
            const double g2 = gx * gx + gy * gy;

            if (g2 < 1e-20) {
                return false;
            }

            double sx = -p * gx / g2, sy = -p * gy / g2;
            const double sl = std::sqrt(sx * sx + sy * sy), cap = 0.5 * rad;

            if (sl > cap) {
                sx *= cap / sl;
                sy *= cap / sl;
            }

            X += sx;
            Y += sy;

            if (std::fabs(p) < 1e-6 && sl < 1e-7 * (vx + vy)) {
                break;
            }
        }

        double gx, gy;

        if (std::fabs(psi(a, b, X, Y, &gx, &gy)) > 1e-4 || std::hypot(X - x0, Y - y0) > rad) {
            return false;
        }

        if (third(a, b, X, Y) >= val(a, X, Y) - 1e-5) {   // a third label is as strong: a junction
            return false;
        }

        x = X;
        y = Y;
        return true;
    }
};

void smooth1d(std::vector<float>& f, int nx, int ny, double sigma, float outside, bool clamp) {
    const int R = std::max(1, static_cast<int>(std::ceil(3 * sigma)));
    std::vector<float> w(2 * R + 1), tmp(f.size());
    float ws = 0;

    for (int d = -R; d <= R; ++d) {
        w[d + R] = static_cast<float>(std::exp(-0.5 * d * d / (sigma * sigma)));
        ws += w[d + R];
    }

    for (float& x : w) {
        x /= ws;
    }

    for (int axis = 0; axis < 2; ++axis) {
        const int len = axis ? ny : nx;
        #pragma omp parallel for schedule(static)

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int c = axis ? j : i;
                float acc = 0;

                for (int d = -R; d <= R; ++d) {
                    int cc = c + d;
                    float v;

                    if (cc < 0 || cc >= len) {
                        if (!clamp) {
                            acc += w[d + R] * outside;
                            continue;
                        }

                        cc = std::min(len - 1, std::max(0, cc));
                    }

                    v = axis ? f[i + static_cast<size_t>(nx) * cc] : f[cc + static_cast<size_t>(nx) * j];
                    acc += w[d + R] * v;
                }

                tmp[i + static_cast<size_t>(nx) * j] = acc;
            }

        f.swap(tmp);
    }
}

// ---- exact Delaunay (Bowyer-Watson) ----------------------------------------------

struct Delaunay {
    std::vector<double> X;          // 2 per point (the last three: the super triangle)
    std::vector<int> T;             // 3 per triangle, ccw
    std::vector<int> N;             // 3 per triangle: the triangle across the edge opposite corner k
    std::vector<char> alive;

    int o2(int a, int b, int c) const {
        return orient2d(X[2 * a], X[2 * a + 1], X[2 * b], X[2 * b + 1], X[2 * c], X[2 * c + 1]);
    }
    int inc(int t, int p) const {
        const int* v = &T[3 * t];
        return incircle(X[2 * v[0]], X[2 * v[0] + 1], X[2 * v[1]], X[2 * v[1] + 1], X[2 * v[2]], X[2 * v[2] + 1],
                        X[2 * p], X[2 * p + 1]);
    }

    // triangulate points P (2 per point); T keeps the triangles of the real points
    void build(const std::vector<double>& P) {
        const int n = static_cast<int>(P.size() / 2);
        double lo[2] = { 1e300, 1e300 }, hi[2] = { -1e300, -1e300 };

        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 2; ++k) {
                lo[k] = std::min(lo[k], P[2 * i + k]);
                hi[k] = std::max(hi[k], P[2 * i + k]);
            }

        const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]);
        const double r = 1000.0 * std::max(1.0, std::max(hi[0] - lo[0], hi[1] - lo[1]));
        X = P;
        X.insert(X.end(), { cx - 2 * r, cy - r, cx + 2 * r, cy - r, cx, cy + 2 * r });
        T.assign({ n, n + 1, n + 2 });
        N.assign({ -1, -1, -1 });
        alive.assign(1, 1);
        std::vector<int> freel, cav, cavb;
        // spatial order (a Morton key on a 1024^2 grid) for short walks
        std::vector<std::pair<uint64_t, int>> ord(n);

        for (int i = 0; i < n; ++i) {
            const uint64_t qx = static_cast<uint64_t>(1023.0 * (P[2 * i] - lo[0]) / std::max(1e-30, hi[0] - lo[0]));
            const uint64_t qy = static_cast<uint64_t>(1023.0 * (P[2 * i + 1] - lo[1]) / std::max(1e-30, hi[1] - lo[1]));
            uint64_t key = 0;

            for (int b = 0; b < 10; ++b) {
                key |= ((qx >> b) & 1) << (2 * b) | ((qy >> b) & 1) << (2 * b + 1);
            }

            ord[i] = { key, i };
        }

        std::sort(ord.begin(), ord.end());
        int t = 0;
        std::vector<char> incav;

        for (const auto& kp : ord) {
            const int p = kp.second;

            if (!alive[t]) {
                t = 0;

                while (!alive[t]) {
                    ++t;
                }
            }

            // walk to the triangle containing p
            for (int guard = 0; guard < 1000000; ++guard) {
                const int* v = &T[3 * t];
                int k;

                for (k = 0; k < 3; ++k)
                    if (o2(v[(k + 1) % 3], v[(k + 2) % 3], p) < 0) {
                        break;
                    }

                if (k == 3) {
                    break;
                }

                t = N[3 * t + k];
            }

            // the cavity: circumcircles strictly containing p
            cav.assign(1, t);
            incav.resize(alive.size(), 0);
            incav[t] = 1;

            for (size_t h = 0; h < cav.size(); ++h)
                for (int k = 0; k < 3; ++k) {
                    const int u = N[3 * cav[h] + k];

                    if (u >= 0 && !incav[u] && inc(u, p) > 0) {
                        incav[u] = 1;
                        cav.push_back(u);
                    }
                }

            // its boundary edges (a, b) ccw, with the outside neighbour
            struct Be {
                int a, b, out;
            };
            std::vector<Be> be;

            for (int c : cav)
                for (int k = 0; k < 3; ++k) {
                    const int u = N[3 * c + k];

                    if (u < 0 || !incav[u]) {
                        be.push_back({ T[3 * c + (k + 1) % 3], T[3 * c + (k + 2) % 3], u });
                    }
                }

            for (int c : cav) {
                alive[c] = 0;
                incav[c] = 0;
                freel.push_back(c);
            }

            std::vector<int> nt(be.size());

            for (size_t e = 0; e < be.size(); ++e) {
                int s;

                if (!freel.empty()) {
                    s = freel.back();
                    freel.pop_back();
                } else {
                    s = static_cast<int>(alive.size());
                    alive.push_back(0);
                    incav.push_back(0);
                    T.insert(T.end(), 3, -1);
                    N.insert(N.end(), 3, -1);
                }

                nt[e] = s;
                alive[s] = 1;
                T[3 * s] = be[e].a;
                T[3 * s + 1] = be[e].b;
                T[3 * s + 2] = p;
                N[3 * s + 2] = be[e].out;   // across (a, b), opposite p

                if (be[e].out >= 0) {   // re-point the outside triangle to s
                    const int u = be[e].out;

                    for (int k = 0; k < 3; ++k) {
                        const int x = T[3 * u + (k + 1) % 3], y = T[3 * u + (k + 2) % 3];

                        if ((x == be[e].b && y == be[e].a) || (x == be[e].a && y == be[e].b)) {
                            N[3 * u + k] = s;
                        }
                    }
                }
            }

            // inside the star: new tri (a, b, p) meets the one starting at b across (b, p)
            // (opposite corner 0 = a) and the one ending at a across (p, a) (corner 1 = b)
            std::unordered_map<int, int> start, end;

            for (size_t e = 0; e < be.size(); ++e) {
                start[be[e].a] = nt[e];
                end[be[e].b] = nt[e];
            }

            for (size_t e = 0; e < be.size(); ++e) {
                N[3 * nt[e]] = start[be[e].b];
                N[3 * nt[e] + 1] = end[be[e].a];
            }

            t = nt[0];
        }

        // keep the triangles of the real points
        std::vector<int> T2;

        for (size_t s = 0; s < alive.size(); ++s)
            if (alive[s] && T[3 * s] < n && T[3 * s + 1] < n && T[3 * s + 2] < n) {
                T2.insert(T2.end(), { T[3 * s], T[3 * s + 1], T[3 * s + 2] });
            }

        T.swap(T2);
    }
};

double tri_quality(const double* a, const double* b, const double* c, double* min_angle = nullptr) {
    const double ux = b[0] - a[0], uy = b[1] - a[1], vx = c[0] - a[0], vy = c[1] - a[1];
    const double area = 0.5 * (ux * vy - uy * vx);
    const double l0 = (b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1]);
    const double l1 = vx * vx + vy * vy, l2 = ux * ux + uy * uy;

    if (min_angle) {
        const double A = std::sqrt(l0), B = std::sqrt(l1), C = std::sqrt(l2);
        auto ang = [](double opp, double s1, double s2) {
            return std::acos(std::max(-1.0, std::min(1.0, (s1 * s1 + s2 * s2 - opp * opp) / (2 * s1 * s2))));
        };
        *min_angle = 180.0 / M_PI * std::min(ang(A, B, C), std::min(ang(B, A, C), ang(C, A, B)));
    }

    return 4.0 * std::sqrt(3.0) * area / std::max(1e-300, l0 + l1 + l2);
}

}  // namespace

bool set_option2d(Mesh2DOptions& o, std::vector<float>& thr, const std::string& name, const std::vector<double>& v) {
    std::string k;

    for (char c : name)
        if (c != '_') {
            k += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

    auto d = [&]() {
        if (v.empty()) {
            throw std::runtime_error("trussnet 2-D: option '" + name + "' needs a numeric value");
        }

        return v[0];
    };

    if (k == "size") {
        o.size = d();
    } else if (k == "hmin") {
        o.hmin = d();
    } else if (k == "hmax") {
        o.hmax = d();
    } else if (k == "k") {
        o.K = d();
    } else if (k == "grad") {
        o.grad = d();
    } else if (k == "sigma") {
        o.sigma = d();
    } else if (k == "graysigma") {
        o.gray_sigma = d();
    } else if (k == "nseed") {
        o.nseed = static_cast<int>(d());
    } else if (k == "iters" || k == "maxiters") {
        o.iters = static_cast<int>(d());
    } else if (k == "fscale") {
        o.fscale = d();
    } else if (k == "fsurf") {
        o.fsurf = d();
    } else if (k == "dt") {
        o.dt = d();
    } else if (k == "snap") {
        o.snap = d();
    } else if (k == "repair") {
        o.repair = static_cast<int>(d());
    } else if (k == "smooth") {
        o.smooth = static_cast<int>(d());
    } else if (k == "verbose") {
        o.verbose = d() != 0;
    } else if (k == "thresholds") {
        thr.assign(v.begin(), v.end());
    } else if (k == "lsize") {
        for (size_t j = 0; j + 1 < v.size(); j += 2) {
            const int l = static_cast<int>(std::lround(v[j]));

            if (l < 0 || l > 65535) {
                throw std::runtime_error("trussnet 2-D: lsize: bad label");
            }

            if (static_cast<int>(o.hlab.size()) <= l) {
                o.hlab.resize(l + 1, 0.0f);
            }

            o.hlab[l] = static_cast<float>(v[j + 1]);
        }
    } else if (k == "gpu" || k == "gpuid" || k == "reratio" || k == "q" || k == "quality" || k == "opt") {
        // (3-D options: accepted and ignored by the 2-D mesher)
    } else {
        return false;
    }

    return true;
}

void mesh2d(Image2D& im, const Mesh2DOptions& o, Mesh2D& out, Mesh2DStats& st) {
    const clk::time_point t0 = clk::now();
    st = Mesh2DStats();
    out = Mesh2D();
    const int nx = im.nx, ny = im.ny;
    const size_t np = static_cast<size_t>(nx) * ny;

    if (nx < 2 || ny < 2) {
        throw std::runtime_error("trussnet 2-D: the image must be at least 2 x 2");
    }

    const bool gray = !im.thresholds.empty();

    if (gray && im.gray.size() != np) {
        throw std::runtime_error("trussnet 2-D: thresholds need a gray-scale image");
    }

    if (!gray && im.lab.size() != np) {
        throw std::runtime_error("trussnet 2-D: no label image");
    }

    Field F;
    F.nx = nx;
    F.ny = ny;
    F.vx = im.vs[0];
    F.vy = im.vs[1];

    // ---- 1. labels and fields
    if (gray) {
        std::vector<float> I(im.gray);
        std::vector<float> T(im.thresholds);
        std::sort(T.begin(), T.end());
        im.thresholds = T;

        if (o.gray_sigma > 0) {
            smooth1d(I, nx, ny, o.gray_sigma, 0.0f, true);
        }

        const int m = static_cast<int>(T.size());
        im.lab.assign(np, 0);

        for (size_t v = 0; v < np; ++v) {
            int l = 0;

            while (l < m && I[v] >= T[l]) {
                ++l;
            }

            im.lab[v] = static_cast<uint16_t>(l);
        }

        // memberships (as the 3-D gray-scale mode): s_k = (I - t_k) / W_k, W_k = 4 x the
        // median intensity step across the iso-line; psi linear in I near each t_k
        std::vector<float> W(m, 1.0f);

        for (int k = 0; k < m; ++k) {
            std::vector<float> steps;

            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t v = i + static_cast<size_t>(nx) * j;

                    if (i + 1 < nx && ((I[v] < T[k]) != (I[v + 1] < T[k]))) {
                        steps.push_back(std::fabs(I[v + 1] - I[v]));
                    }

                    if (j + 1 < ny && ((I[v] < T[k]) != (I[v + nx] < T[k]))) {
                        steps.push_back(std::fabs(I[v + nx] - I[v]));
                    }
                }

            if (!steps.empty()) {
                std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
                W[k] = std::max(1e-12f, 4.0f * steps[steps.size() / 2]);
            }
        }

        F.nlab = m + 1;
        F.phi.assign(static_cast<size_t>(F.nlab) * np, 0.0f);

        for (int l = 0; l <= m; ++l)
            for (size_t v = 0; v < np; ++v) {
                float mm = 1e30f;

                if (l >= 1) {
                    mm = std::min(mm, (I[v] - T[l - 1]) / W[l - 1]);
                }

                if (l < m) {
                    mm = std::min(mm, (T[l] - I[v]) / W[l]);
                }

                F.phi[static_cast<size_t>(l) * np + v] = std::min(1.0f, std::max(0.0f, 0.5f + mm));
            }
    } else {
        int m = 0;

        for (uint16_t l : im.lab) {
            m = std::max<int>(m, l);
        }

        F.nlab = m + 1;
        F.phi.assign(static_cast<size_t>(F.nlab) * np, 0.0f);
        #pragma omp parallel for schedule(dynamic, 1)

        for (int l = 0; l < F.nlab; ++l) {
            std::vector<float> f(np);

            for (size_t v = 0; v < np; ++v) {
                f[v] = im.lab[v] == l ? 1.0f : 0.0f;
            }

            if (o.sigma > 0) {
                smooth1d(f, nx, ny, o.sigma, l == 0 ? 1.0f : 0.0f, false);
            }

            std::copy(f.begin(), f.end(), F.phi.begin() + static_cast<size_t>(l) * np);
        }
    }

    if (F.nlab < 2) {
        throw std::runtime_error("trussnet 2-D: the image has no non-zero label (0 = exterior)");
    }

    // ---- 2. sizing per pixel
    const double vmin = std::min(F.vx, F.vy);
    const double hbase = o.size > 0 ? o.size : 3 * vmin;
    double hmin = o.hmin > 0 ? o.hmin : hbase / 3, hmax = o.hmax > 0 ? o.hmax : hbase;

    for (float v : o.hlab)
        if (v > 0) {
            hmin = std::min(hmin, static_cast<double>(v));
            hmax = std::max(hmax, static_cast<double>(v));
        }

    const bool user = o.hvox.size() == np;

    if (!o.hvox.empty() && !user) {
        throw std::runtime_error("trussnet 2-D: the sizing field must have one value per pixel");
    }

    if (user)
        for (float v : o.hvox)
            if (v > 0) {
                hmin = std::min(hmin, static_cast<double>(v));
                hmax = std::max(hmax, static_cast<double>(v));
            }

    std::vector<float> H(np);
    auto hl = [&](int l) {
        return l < static_cast<int>(o.hlab.size()) && o.hlab[l] > 0 ? o.hlab[l] : 0.0f;
    };
    #pragma omp parallel for schedule(static)

    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const size_t v = i + static_cast<size_t>(nx) * j;
            const int a = im.lab[v];
            double h = hl(a) > 0 ? hl(a) : hbase;

            if (a == 0) {
                h = hmax;
            } else {
                // the strongest competitor at the pixel centre
                int b = -1;
                double pb = -1e30;

                for (int l = 0; l < F.nlab; ++l)
                    if (l != a && F.at(l, i, j) > pb) {
                        pb = F.at(l, i, j);
                        b = l;
                    }

                if (b >= 0 && F.at(a, i, j) - pb < 0.45) {
                    if (hl(b) > 0) {
                        h = std::min(h, static_cast<double>(hl(b)));
                    }

                    // curvature of the level set of psi = phi_a - phi_b (central differences)
                    auto ps = [&](int di, int dj) {
                        return F.at(a, i + di, j + dj) - F.at(b, i + di, j + dj);
                    };
                    const double px = (ps(1, 0) - ps(-1, 0)) / (2 * F.vx), py = (ps(0, 1) - ps(0, -1)) / (2 * F.vy);
                    const double pxx = (ps(1, 0) - 2 * ps(0, 0) + ps(-1, 0)) / (F.vx * F.vx);
                    const double pyy = (ps(0, 1) - 2 * ps(0, 0) + ps(0, -1)) / (F.vy * F.vy);
                    const double pxy = (ps(1, 1) - ps(1, -1) - ps(-1, 1) + ps(-1, -1)) / (4 * F.vx * F.vy);
                    const double g2 = px * px + py * py;

                    if (g2 > 1e-12) {
                        const double kap = std::fabs(pxx * py * py - 2 * px * py * pxy + pyy * px * px) / (g2 * std::sqrt(g2));

                        if (kap > 0) {
                            h = std::min(h, 1.0 / (o.K * kap));
                        }
                    }
                }
            }

            if (user && o.hvox[v] > 0) {
                h = o.hvox[v];
            }

            H[v] = static_cast<float>(std::min(hmax, std::max(hmin, h)));
        }

    // gradient limit: h(p) <= h(q) + grad |p - q| (raster sweeps until nothing changes)
    for (int pass = 0; pass < 200; ++pass) {
        bool changed = false;

        for (int dir = 0; dir < 2; ++dir)
            for (int jj = 0; jj < ny; ++jj)
                for (int ii = 0; ii < nx; ++ii) {
                    const int i = dir ? nx - 1 - ii : ii, j = dir ? ny - 1 - jj : jj;
                    float& h = H[i + static_cast<size_t>(nx) * j];

                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            const int a = i + di, b = j + dj;

                            if ((!di && !dj) || a < 0 || b < 0 || a >= nx || b >= ny) {
                                continue;
                            }

                            const float lim = H[a + static_cast<size_t>(nx) * b] +
                                              static_cast<float>(o.grad * std::hypot(di * F.vx, dj * F.vy));

                            if (h > lim + 1e-6f) {
                                h = lim;
                                changed = true;
                            }
                        }
                }

        if (!changed) {
            break;
        }
    }

    auto h_at = [&](double x, double y) {
        const int i = std::min(nx - 1, std::max(0, static_cast<int>(std::lround(x / F.vx))));
        const int j = std::min(ny - 1, std::max(0, static_cast<int>(std::lround(y / F.vy))));
        return static_cast<double>(H[i + static_cast<size_t>(nx) * j]);
    };
    st.ms_fields = ms_since(t0);

    // ---- 3. seeding: graded hexagonal lattices, junction nodes
    const clk::time_point t1 = clk::now();
    std::vector<double> P;
    std::vector<int> lab, pa, pb;
    std::vector<uint8_t> typ;
    auto add = [&](double x, double y, int l, int t, int a, int b) {
        P.push_back(x);
        P.push_back(y);
        lab.push_back(l);
        typ.push_back(static_cast<uint8_t>(t));
        pa.push_back(a);
        pb.push_back(b);
    };
    const double W = (nx - 1) * F.vx, Hh = (ny - 1) * F.vy;

    // junctions: 2 x 2 pixel blocks with >= 3 labels -> solve phi_a = phi_b = phi_c
    for (int j = -1; j < ny; ++j)
        for (int i = -1; i < nx; ++i) {
            int ls[4], n = 0;

            for (int k = 0; k < 4; ++k) {
                const int a = i + (k & 1), b = j + (k >> 1);
                const int l = (a < 0 || b < 0 || a >= nx || b >= ny) ? 0 : im.lab[a + static_cast<size_t>(nx) * b];
                bool seen = false;

                for (int q = 0; q < n; ++q) {
                    seen |= ls[q] == l;
                }

                if (!seen) {
                    ls[n++] = l;
                }
            }

            if (n < 3) {
                continue;
            }

            double x = (i + 0.5) * F.vx, y = (j + 0.5) * F.vy;
            // the top three fields at the block centre
            int a = ls[0], b = ls[1], c = ls[2];
            {
                std::vector<std::pair<double, int>> v;

                for (int q = 0; q < n; ++q) {
                    v.push_back({ -F.val(ls[q], x, y), ls[q] });
                }

                std::sort(v.begin(), v.end());
                a = v[0].second;
                b = v[1].second;
                c = v[2].second;
            }
            bool ok = false;

            for (int it = 0; it < 15; ++it) {
                double g1x, g1y, g2x, g2y;
                const double f1 = F.psi(a, b, x, y, &g1x, &g1y), f2 = F.psi(a, c, x, y, &g2x, &g2y);
                const double det = g1x * g2y - g1y * g2x;

                if (std::fabs(det) < 1e-20) {
                    break;
                }

                x -= (f1 * g2y - f2 * g1y) / det;
                y -= (g1x * f2 - g2x * f1) / det;

                if (std::fabs(f1) < 1e-6 && std::fabs(f2) < 1e-6) {
                    ok = true;
                    break;
                }
            }

            if (!ok || std::hypot(x - (i + 0.5) * F.vx, y - (j + 0.5) * F.vy) > 1.5 * vmin) {
                continue;
            }

            if (x < -0.5 * F.vx || y < -0.5 * F.vy || x > W + 0.5 * F.vx || y > Hh + 0.5 * F.vy) {
                continue;
            }

            bool near = false;   // one node per junction

            for (size_t q = 0; q < typ.size() && !near; ++q) {
                near = std::hypot(P[2 * q] - x, P[2 * q + 1] - y) < 0.5 * h_at(x, y);
            }

            if (!near) {
                // own label: a non-exterior one
                const int own = a ? a : (b ? b : c);
                const int o1 = own == a ? b : a, o2 = own == c ? b : c;
                add(x, y, own, T_JUNCTION, o1, o2);
            }
        }

    st.junctions = typ.size();
    // lattices: level k keeps the points whose size quantizes to k
    const int L = std::max(1, o.nseed);
    const double ratio = hmax > hmin ? std::log(hmax / hmin) / std::max(1, L - 1) : 0.0;
    auto level = [&](double h) {
        return ratio > 0 ? std::min(L - 1, std::max(0, static_cast<int>(std::lround(std::log(h / hmin) / ratio)))) : 0;
    };
    const size_t nj = typ.size();
    const double jcell = std::max(1e-9, 0.5 * hmax);
    std::unordered_map<int64_t, std::vector<int>> jbins;   // (the junction clearance)
    auto jkey = [&](double x, double y) {
        return (static_cast<int64_t>(std::floor(x / jcell)) << 32) ^ static_cast<int64_t>(std::floor(y / jcell) + 1e6);
    };

    for (size_t q = 0; q < nj; ++q) {
        jbins[jkey(P[2 * q], P[2 * q + 1])].push_back(static_cast<int>(q));
    }

    for (int k = 0; k < (ratio > 0 ? L : 1); ++k) {
        const double s = hmin * std::exp(ratio * k);
        const double dy = s * std::sqrt(3.0) / 2;

        for (int r = 0; r * dy <= Hh + 0.5 * F.vy; ++r)
            for (int c = 0; (c + 0.5 * (r & 1)) * s <= W + 0.5 * F.vx; ++c) {
                const double x = (c + 0.5 * (r & 1)) * s, y = r * dy;
                const double h = h_at(x, y);

                if (level(h) != k) {
                    continue;
                }

                const int l = F.label(x, y);

                if (l == 0) {
                    continue;
                }

                bool near = false;

                for (int dj = -1; dj <= 1 && !near; ++dj)
                    for (int di = -1; di <= 1 && !near; ++di) {
                        auto itb = jbins.find(jkey(x + di * jcell, y + dj * jcell));

                        if (itb != jbins.end())
                            for (int q : itb->second)
                                if (std::hypot(P[2 * q] - x, P[2 * q + 1] - y) < 0.5 * h) {
                                    near = true;
                                    break;
                                }
                    }

                if (!near) {
                    add(x, y, l, T_INTERIOR, -1, -1);
                }
            }
    }

    st.seeds = typ.size();
    int n = static_cast<int>(typ.size());

    // ---- 4. relaxation
    const double tnb = 1.8;   // neighbour search radius / h
    const double cell = std::max(1e-9, tnb * hbase);
    std::vector<int> nbr, nnb;
    const int K = 16;
    auto build_nbrs = [&]() {
        std::unordered_map<int64_t, std::vector<int>> g;
        auto key = [&](double x, double y) {
            return (static_cast<int64_t>(std::floor(x / cell)) << 32) ^ static_cast<int64_t>(std::floor(y / cell) + 1e6);
        };

        for (int i = 0; i < n; ++i) {
            g[key(P[2 * i], P[2 * i + 1])].push_back(i);
        }

        nbr.assign(static_cast<size_t>(n) * K, -1);
        nnb.assign(n, 0);
        #pragma omp parallel for schedule(dynamic, 256)

        for (int i = 0; i < n; ++i) {
            const double x = P[2 * i], y = P[2 * i + 1], hi = h_at(x, y), R = tnb * hi;
            const int m = static_cast<int>(std::ceil(R / cell));
            std::vector<std::pair<double, int>> c;

            for (int dj = -m; dj <= m; ++dj)
                for (int di = -m; di <= m; ++di) {
                    auto it = g.find(key(x + di * cell, y + dj * cell));

                    if (it == g.end()) {
                        continue;
                    }

                    for (int j : it->second) {
                        if (j == i) {
                            continue;
                        }

                        if (typ[i] == T_INTERIOR && typ[j] == T_INTERIOR && lab[i] != lab[j]) {
                            continue;
                        }

                        const double d = std::hypot(P[2 * j] - x, P[2 * j + 1] - y);

                        if (d < R) {
                            c.push_back({ d, j });
                        }
                    }
                }

            std::sort(c.begin(), c.end());
            const int k = std::min<int>(K, static_cast<int>(c.size()));

            for (int q = 0; q < k; ++q) {
                nbr[static_cast<size_t>(i) * K + q] = c[q].second;
            }

            nnb[i] = k;
        }
    };
    // trap an interior node at (x, y): project it onto the interface it is near; the
    // new type / partner go to typ2 / pa2 (Jacobi: the neighbours read the old ones)
    std::vector<uint8_t> typ2;
    std::vector<int> pa2;
    // returns the (possibly projected) position, ok = false: reject the move
    auto trap = [&](int i, Moved m) {
        double x = m.x, y = m.y;
        int sec;
        double mg;
        const int l = F.label(x, y, &sec, &mg);
        const double h = h_at(x, y);

        if (l != lab[i]) {   // crossed into l: onto lab|l
            double X = x, Y = y;

            if (F.project(lab[i], l, X, Y, 0.6 * h)) {
                typ2[i] = T_INTERFACE;
                pa2[i] = l;
                return Moved{ true, X, Y };
            }

            return Moved{ false, x, y };   // reject the move
        }

        if (sec >= 0 && mg < 0.45) {
            double gx, gy;
            const double p = F.psi(lab[i], sec, x, y, &gx, &gy);
            const double g = std::hypot(gx, gy);

            if (g > 1e-12 && std::fabs(p) / g < o.snap * h) {
                double X = x, Y = y;

                if (F.project(lab[i], sec, X, Y, o.snap * h * 1.5)) {
                    typ2[i] = T_INTERFACE;
                    pa2[i] = sec;
                    return Moved{ true, X, Y };
                }
            }
        }

        return Moved{ true, x, y };
    };

    typ2 = typ;
    pa2 = pa;

    for (int i = 0; i < n; ++i)   // seeds near an interface start on it
        if (typ[i] == T_INTERIOR) {
            const Moved m = trap(i, Moved{ true, P[2 * i], P[2 * i + 1] });
            P[2 * i] = m.x;
            P[2 * i + 1] = m.y;
        }

    typ = typ2;
    pa = pa2;

    std::vector<double> P2(P);
    // interface density control: the snapping keeps adding nodes to a curve of fixed
    // length, which purely repulsive springs cannot thin out (crowded boundary
    // nodes -> flat triangles against the curve): drop an interface node within
    // 0.6 h of another node on the same interface, or of a junction
    auto thin_interfaces = [&]() {
        std::vector<char> drop(n, 0);
        size_t nd = 0;

        for (int i = 0; i < n; ++i) {
            if (typ[i] != T_INTERFACE) {
                continue;
            }

            const double hi = h_at(P[2 * i], P[2 * i + 1]);

            for (int q = 0; q < nnb[i]; ++q) {
                const int j = nbr[static_cast<size_t>(i) * K + q];

                if (drop[j]) {
                    continue;
                }

                const bool same = typ[j] == T_INTERFACE &&
                                  ((lab[j] == lab[i] && pa[j] == pa[i]) || (lab[j] == pa[i] && pa[j] == lab[i]));
                const bool junc = typ[j] == T_JUNCTION;

                if ((same && j < i) || junc) {
                    const double d = std::hypot(P[2 * j] - P[2 * i], P[2 * j + 1] - P[2 * i + 1]);

                    if (d < 0.6 * 0.5 * (hi + h_at(P[2 * j], P[2 * j + 1]))) {
                        drop[i] = 1;
                        ++nd;
                        break;
                    }
                }
            }
        }

        if (!nd) {
            return;
        }

        int w = 0;

        for (int i = 0; i < n; ++i) {
            if (drop[i]) {
                continue;
            }

            P[2 * w] = P[2 * i];
            P[2 * w + 1] = P[2 * i + 1];
            lab[w] = lab[i];
            typ[w] = typ[i];
            pa[w] = pa[i];
            pb[w] = pb[i];
            ++w;
        }

        n = w;
        P.resize(2 * n);
        lab.resize(n);
        typ.resize(n);
        pa.resize(n);
        pb.resize(n);
        P2.resize(2 * n);
        typ2 = typ;
        pa2 = pa;
        build_nbrs();
    };
    int it = 0;

    for (; it < o.iters; ++it) {
        if (it % 5 == 0) {
            build_nbrs();
        }

        if (it % 20 == 10) {
            thin_interfaces();
        }

        double maxmove = 0;
        #pragma omp parallel for schedule(dynamic, 256) reduction(max : maxmove)

        for (int i = 0; i < n; ++i) {
            double x = P[2 * i], y = P[2 * i + 1];
            P2[2 * i] = x;
            P2[2 * i + 1] = y;

            if (typ[i] == T_JUNCTION) {
                continue;
            }

            const double hi = h_at(x, y);
            double fx = 0, fy = 0;

            for (int q = 0; q < nnb[i]; ++q) {
                const int j = nbr[static_cast<size_t>(i) * K + q];
                const double ex = x - P[2 * j], ey = y - P[2 * j + 1], l = std::hypot(ex, ey);
                const double l0 = ((typ[i] != T_INTERIOR && typ[j] != T_INTERIOR) ? o.fsurf : o.fscale) * 0.5 *
                                  (hi + h_at(P[2 * j], P[2 * j + 1]));

                if (l < l0 && l > 1e-12) {
                    fx += (l0 - l) / l * ex;
                    fy += (l0 - l) / l * ey;
                }
            }

            double dx = o.dt * fx, dy = o.dt * fy;
            const double dl = std::hypot(dx, dy), cap = 0.2 * hi;

            if (dl > cap) {
                dx *= cap / dl;
                dy *= cap / dl;
            }

            if (typ[i] == T_INTERIOR) {
                const Moved m = trap(i, Moved{ true, x + dx, y + dy });

                if (m.ok) {
                    P2[2 * i] = m.x;
                    P2[2 * i + 1] = m.y;
                }
            } else {   // glide: the tangential part, then back onto the curve
                double gx, gy;
                F.psi(lab[i], pa[i], x, y, &gx, &gy);
                const double g = std::hypot(gx, gy);

                if (g < 1e-12) {
                    continue;
                }

                const double tx = -gy / g, ty = gx / g, s = dx * tx + dy * ty;
                double X = x + s * tx, Y = y + s * ty;

                if (F.project(lab[i], pa[i], X, Y, 0.5 * hi)) {
                    P2[2 * i] = X;
                    P2[2 * i + 1] = Y;
                }
            }

            maxmove = std::max(maxmove, std::hypot(P2[2 * i] - x, P2[2 * i + 1] - y) / hi);
        }

        P.swap(P2);
        typ = typ2;
        pa = pa2;

        if (o.verbose && it % 50 == 0) {
            TN_FPRINTF(stderr, "[2d] iter %d: max move %.4g h\n", it, maxmove);
        }

        if (maxmove < 1e-3) {
            ++it;
            break;
        }
    }

    st.iterations = it;
    st.ms_relax = ms_since(t1);

    // ---- 5. Delaunay, labels, repair
    const clk::time_point t2 = clk::now();
    Delaunay D;
    std::vector<int> tl;
    std::vector<char> span;
    auto has = [&](int v, int l) {
        return lab[v] == l || (typ[v] >= T_INTERFACE && pa[v] == l) || (typ[v] == T_JUNCTION && pb[v] == l);
    };
    auto triangulate = [&]() {
        // (exact duplicates would break the exact predicates' assumptions: drop them)
        std::unordered_map<uint64_t, int> seen;
        std::vector<int> keep;

        for (int i = 0; i < n; ++i) {
            uint64_t a, b;
            std::memcpy(&a, &P[2 * i], 8);
            std::memcpy(&b, &P[2 * i + 1], 8);
            const uint64_t k = a * 0x9E3779B97F4A7C15ULL ^ b;

            if (seen.emplace(k, i).second) {
                keep.push_back(i);
            }
        }

        if (static_cast<int>(keep.size()) < n) {
            std::vector<double> P3;
            std::vector<int> l3, a3, b3;
            std::vector<uint8_t> t3;

            for (int i : keep) {
                P3.insert(P3.end(), { P[2 * i], P[2 * i + 1] });
                l3.push_back(lab[i]);
                a3.push_back(pa[i]);
                b3.push_back(pb[i]);
                t3.push_back(typ[i]);
            }

            P.swap(P3);
            lab.swap(l3);
            pa.swap(a3);
            pb.swap(b3);
            typ.swap(t3);
            n = static_cast<int>(keep.size());
        }

        D.build(P);
        const int nt = static_cast<int>(D.T.size() / 3);
        tl.assign(nt, 0);
        span.assign(nt, 0);
        #pragma omp parallel for schedule(static)

        for (int t = 0; t < nt; ++t) {
            const int* v = &D.T[3 * t];
            const double cx = (P[2 * v[0]] + P[2 * v[1]] + P[2 * v[2]]) / 3, cy = (P[2 * v[0] + 1] + P[2 * v[1] + 1] + P[2 * v[2] + 1]) / 3;
            const int cl = F.label(cx, cy);
            int cands[3] = { lab[v[0]], typ[v[0]] ? pa[v[0]] : -1, typ[v[0]] == T_JUNCTION ? pb[v[0]] : -1 };
            int common = -1, ncommon = 0;

            for (int c : cands) {
                if (c < 0 || !has(v[1], c) || !has(v[2], c)) {
                    continue;
                }

                ++ncommon;

                if (common < 0 || c == cl) {
                    common = c;
                }
            }

            if (ncommon == 0) {
                tl[t] = cl;
                span[t] = 1;
            } else {
                tl[t] = common;
            }
        }
    };

    triangulate();
    const double guard = 0.25;

    for (int round = 0; round < o.repair; ++round) {
        // the triangle adjacency of the kept mesh (edges keyed by vertex pairs)
        const int nt = static_cast<int>(D.T.size() / 3);
        std::unordered_map<uint64_t, int> edge;
        std::vector<int> adj(3 * static_cast<size_t>(nt), -1);

        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < 3; ++k) {
                const int a = D.T[3 * t + (k + 1) % 3], b = D.T[3 * t + (k + 2) % 3];
                const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) | static_cast<uint32_t>(std::max(a, b));
                auto ins = edge.emplace(key, 3 * t + k);

                if (!ins.second) {
                    adj[3 * t + k] = ins.first->second / 3;
                    adj[ins.first->second] = t;
                }
            }

        struct Fix {
            double x, y;
            int a, b;
        };

        std::vector<Fix> fixes;

        size_t bad = 0, nspan = 0, rej_proj = 0, rej_near = 0;

        for (int t = 0; t < nt; ++t) {
            const int* v = &D.T[3 * t];

            for (int k = 0; k < 3; ++k) {
                const int u = adj[3 * t + k];
                const int la = tl[t], lb = u >= 0 ? tl[u] : 0;

                if (la == lb || (u >= 0 && la < lb) || (la == 0 && u < 0)) {
                    continue;   // (each interface edge once, from its larger label)
                }

                const int e0 = v[(k + 1) % 3], e1 = v[(k + 2) % 3];

                if (has(e0, la) && has(e0, lb) && has(e1, la) && has(e1, lb)) {
                    continue;
                }

                ++bad;
                double x = 0.5 * (P[2 * e0] + P[2 * e1]), y = 0.5 * (P[2 * e0 + 1] + P[2 * e1 + 1]);
                const double el = std::hypot(P[2 * e0] - P[2 * e1], P[2 * e0 + 1] - P[2 * e1 + 1]);

                if (F.project(la, lb, x, y, 0.6 * el)) {
                    fixes.push_back({ x, y, la, lb });
                } else {
                    ++rej_proj;
                }
            }

            if (span[t]) {
                ++nspan;

                // an edge whose end labels differ: its first label change, onto that interface
                for (int k = 0; k < 3; ++k) {
                    const int p = v[k], q = v[(k + 1) % 3];
                    double ax = P[2 * p], ay = P[2 * p + 1], bx = P[2 * q], by = P[2 * q + 1];
                    int la = F.label(ax, ay);
                    int lb2 = la;
                    double sx = ax, sy = ay;

                    for (int s = 1; s <= 16; ++s) {
                        const double f = s / 16.0;
                        const double x = ax + f * (bx - ax), y = ay + f * (by - ay);
                        lb2 = F.label(x, y);

                        if (lb2 != la) {
                            // bisect (sx, sy) [la] .. (x, y) [lb2]
                            double x0 = sx, y0 = sy, x1 = x, y1 = y;

                            for (int b = 0; b < 30; ++b) {
                                const double mx = 0.5 * (x0 + x1), my = 0.5 * (y0 + y1);

                                if (F.label(mx, my) == la) {
                                    x0 = mx;
                                    y0 = my;
                                } else {
                                    x1 = mx;
                                    y1 = my;
                                }
                            }

                            double fx = 0.5 * (x0 + x1), fy = 0.5 * (y0 + y1);

                            if (F.project(la, lb2, fx, fy, 0.3 * h_at(fx, fy))) {
                                fixes.push_back({ fx, fy, la, lb2 });
                            }

                            break;
                        }

                        sx = x;
                        sy = y;
                    }
                }
            }
        }

        st.bad_edges = bad;
        st.spanning = nspan;

        if (fixes.empty()) {
            break;
        }

        // spacing guard against the nodes and the other fixes
        std::unordered_map<int64_t, std::vector<int>> g;
        const double c2 = std::max(1e-9, 0.5 * hmin);
        auto key = [&](double x, double y) {
            return (static_cast<int64_t>(std::floor(x / c2)) << 32) ^ static_cast<int64_t>(std::floor(y / c2) + 1e6);
        };

        for (int i = 0; i < n; ++i) {
            g[key(P[2 * i], P[2 * i + 1])].push_back(i);
        }

        size_t added = 0, promoted = 0;

        for (const Fix& f : fixes) {
            const double r = guard * h_at(f.x, f.y);
            const int m = static_cast<int>(std::ceil(r / c2));
            int near = -1;
            double dn = 1e300;

            for (int dj = -m; dj <= m; ++dj)
                for (int di = -m; di <= m; ++di) {
                    auto itb = g.find(key(f.x + di * c2, f.y + dj * c2));

                    if (itb == g.end()) {
                        continue;
                    }

                    for (int j : itb->second) {
                        const double d = std::hypot(P[2 * j] - f.x, P[2 * j + 1] - f.y);
                        // a node on another interface (two interfaces ~ a pixel apart in
                        // a thin layer) only needs 0.35 pixel clear; the rest 0.25 h
                        const bool other = typ[j] != T_INTERIOR && !(has(j, f.a) && has(j, f.b));

                        if (d < (other ? std::min(r, 0.35 * vmin) : r) && d < dn) {
                            dn = d;
                            near = j;
                        }
                    }
                }

            if (near >= 0) {
                // an interior node of one of the two labels right next to the interface
                // point: put that node on the interface instead (a promotion)
                const int j = near;

                if (typ[j] == T_INTERIOR && (lab[j] == f.a || lab[j] == f.b)) {
                    double X = P[2 * j], Y = P[2 * j + 1];

                    if (F.project(f.a, f.b, X, Y, 0.5 * h_at(X, Y))) {
                        P[2 * j] = X;
                        P[2 * j + 1] = Y;
                        typ[j] = T_INTERFACE;
                        pa[j] = lab[j] == f.a ? f.b : f.a;
                        ++added;
                        ++promoted;
                        continue;
                    }
                }

                ++rej_near;
                continue;
            }

            const int own = f.a ? f.a : f.b, oth = f.a ? f.b : f.a;
            add(f.x, f.y, own, T_INTERFACE, oth, -1);
            g[key(f.x, f.y)].push_back(n);
            ++n;
            ++added;
        }

        st.repairs += added;
        st.repair_rounds = round + 1;

        if (std::getenv("TN_2D_DEBUG")) {
            std::fprintf(stderr, "[2d] repair %d: %zu bad edges, %zu spanning; %zu fixes (edge projection failed %zu), "
                         "%zu too near, %zu added (%zu promoted)\n", round, bad, nspan, fixes.size(), rej_proj, rej_near, added, promoted);
        }

        if (!added) {
            break;
        }

        triangulate();
    }

    // flat caps on a curve: a triangle whose three nodes are all on one interface a|b
    // (consecutive nodes of a slightly wavy curve) is legal Delaunay but can be nearly
    // flat; flip its longest edge with the triangle across it (the two new triangles
    // take that neighbour's label; the curve nodes stay on the curve: conforming)
    auto cap_flips = [&]() {
        const int nt = static_cast<int>(D.T.size() / 3);
        std::unordered_map<uint64_t, int> edge;
        std::vector<int> adj(3 * static_cast<size_t>(nt), -1);

        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < 3; ++k) {
                const int a = D.T[3 * t + (k + 1) % 3], b = D.T[3 * t + (k + 2) % 3];
                const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) | static_cast<uint32_t>(std::max(a, b));
                auto ins = edge.emplace(key, 3 * t + k);

                if (!ins.second) {
                    adj[3 * t + k] = ins.first->second / 3;
                    adj[ins.first->second] = t;
                }
            }

        std::vector<char> used(nt, 0);
        size_t nflip = 0, dbg[5] = { 0, 0, 0, 0, 0 };

        for (int t = 0; t < nt; ++t) {
            int* v = &D.T[3 * t];

            if (used[t] || tl[t] == 0 || tri_quality(&P[2 * v[0]], &P[2 * v[1]], &P[2 * v[2]]) > 0.2) {
                continue;
            }

            ++dbg[0];

            // one interface shared by the three nodes
            bool cap = false;
            const int c0[3] = { lab[v[0]], typ[v[0]] ? pa[v[0]] : -1, typ[v[0]] == T_JUNCTION ? pb[v[0]] : -1 };

            for (int x = 0; x < 3 && !cap; ++x)
                for (int y = x + 1; y < 3 && !cap; ++y) {
                    if (c0[x] < 0 || c0[y] < 0) {
                        continue;
                    }

                    cap = has(v[1], c0[x]) && has(v[1], c0[y]) && has(v[2], c0[x]) && has(v[2], c0[y]);
                }

            if (!cap) {
                continue;
            }

            ++dbg[1];
            int capa = -1, capb = -1;   // the shared interface

            for (int x = 0; x < 3 && capa < 0; ++x)
                for (int y = x + 1; y < 3; ++y)
                    if (c0[x] >= 0 && c0[y] >= 0 && has(v[1], c0[x]) && has(v[1], c0[y]) && has(v[2], c0[x]) &&
                            has(v[2], c0[y])) {
                        capa = c0[x];
                        capb = c0[y];
                        break;
                    }

            int k = 0;   // the longest edge: opposite corner k
            double best = -1;

            for (int q = 0; q < 3; ++q) {
                const int a = v[(q + 1) % 3], b = v[(q + 2) % 3];
                const double l = std::hypot(P[2 * a] - P[2 * b], P[2 * a + 1] - P[2 * b + 1]);

                if (l > best) {
                    best = l;
                    k = q;
                }
            }

            const int u = adj[3 * t + k];

            if (u < 0) {   // on the convex hull: a cap against the exterior -> dropped
                if (capa == 0 || capb == 0) {
                    tl[t] = 0;
                    used[t] = 1;
                    ++nflip;
                }

                continue;
            }

            if (used[u]) {
                continue;
            }

            ++dbg[2];

            int* w = &D.T[3 * u];
            const int pk = v[k], a = v[(k + 1) % 3], b = v[(k + 2) % 3];
            int qv = -1;

            for (int q = 0; q < 3; ++q)
                if (w[q] != a && w[q] != b) {
                    qv = w[q];
                }

            // t = (pk, a, b) is ccw and q lies across (a, b): the flip is valid iff the
            // quad pk, a, q, b is convex, i.e. both new triangles are ccw
            if (qv < 0 || D.o2(pk, a, qv) <= 0 || D.o2(pk, qv, b) <= 0) {
                continue;
            }

            ++dbg[3];

            const int t1[3] = { pk, a, qv }, t2[3] = { pk, qv, b };
            const int lu = tl[u];
            std::copy(t1, t1 + 3, v);
            std::copy(t2, t2 + 3, w);
            tl[t] = tl[u] = lu;
            span[t] = span[u] = 0;
            used[t] = used[u] = 1;
            ++nflip;
        }

        if (std::getenv("TN_2D_DEBUG")) {
            std::fprintf(stderr, "[2d] caps: flat %zu, one interface %zu, neighbour %zu, convex %zu, flipped %zu\n", dbg[0],
                         dbg[1], dbg[2], dbg[3], nflip);
        }

        return nflip;
    };

    for (int pass = 0; pass < 3 && cap_flips(); ++pass) {
    }

    // the final conformity count
    {
        const int nt = static_cast<int>(D.T.size() / 3);
        std::unordered_map<uint64_t, int> edge;
        std::vector<int> adj(3 * static_cast<size_t>(nt), -1);

        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < 3; ++k) {
                const int a = D.T[3 * t + (k + 1) % 3], b = D.T[3 * t + (k + 2) % 3];
                const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) | static_cast<uint32_t>(std::max(a, b));
                auto ins = edge.emplace(key, 3 * t + k);

                if (!ins.second) {
                    adj[3 * t + k] = ins.first->second / 3;
                    adj[ins.first->second] = t;
                }
            }

        size_t bad = 0, nspan = 0;

        for (int t = 0; t < nt; ++t) {
            nspan += span[t] && tl[t] != 0;

            for (int k = 0; k < 3; ++k) {
                const int u = adj[3 * t + k];
                const int la = tl[t], lb = u >= 0 ? tl[u] : 0;

                if (la == lb || (u >= 0 && la < lb) || (la == 0 && u < 0)) {
                    continue;
                }

                const int e0 = D.T[3 * t + (k + 1) % 3], e1 = D.T[3 * t + (k + 2) % 3];
                bad += !(has(e0, la) && has(e0, lb) && has(e1, la) && has(e1, lb));
            }
        }

        st.bad_edges = bad;
        st.spanning = nspan;

        // ---- 6. guarded smoothing of the interior nodes (the connectivity is kept)
        std::vector<std::vector<int>> vt(n);

        for (int t = 0; t < nt; ++t)
            if (tl[t] != 0)
                for (int k = 0; k < 3; ++k) {
                    vt[D.T[3 * t + k]].push_back(t);
                }

        auto worst = [&](int v, double x, double y) {
            double q = 1e30;
            const double sx = P[2 * v], sy = P[2 * v + 1];
            P[2 * v] = x;
            P[2 * v + 1] = y;

            for (int t : vt[v]) {
                const int* w = &D.T[3 * t];
                q = std::min(q, tri_quality(&P[2 * w[0]], &P[2 * w[1]], &P[2 * w[2]]));
            }

            P[2 * v] = sx;
            P[2 * v + 1] = sy;
            return q;
        };

        for (int pass = 0; pass < o.smooth; ++pass)
            for (int v = 0; v < n; ++v) {
                if (typ[v] != T_INTERIOR || vt[v].empty()) {
                    continue;
                }

                double sx = 0, sy = 0;
                int c = 0;

                for (int t : vt[v])
                    for (int k = 0; k < 3; ++k) {
                        const int w = D.T[3 * t + k];

                        if (w != v) {
                            sx += P[2 * w];
                            sy += P[2 * w + 1];
                            ++c;
                        }
                    }

                const double x = sx / c, y = sy / c;

                if (F.label(x, y) != lab[v]) {
                    continue;
                }

                if (worst(v, x, y) > worst(v, P[2 * v], P[2 * v + 1])) {
                    P[2 * v] = x;
                    P[2 * v + 1] = y;
                }
            }

        // ---- 7. output: the kept triangles, compacted nodes, boundary / interface edges
        std::vector<int> remap(n, -1);
        std::vector<double> q;
        q.reserve(nt);
        double amin = 180;
        st.label_area.assign(F.nlab, 0.0);

        for (int t = 0; t < nt; ++t) {
            if (tl[t] == 0) {
                continue;
            }

            const int* w = &D.T[3 * t];

            for (int k = 0; k < 3; ++k) {
                if (remap[w[k]] < 0) {
                    remap[w[k]] = static_cast<int>(out.node.size() / 2);
                    const double u = P[2 * w[k]] / F.vx, vv = P[2 * w[k] + 1] / F.vy;
                    out.node.push_back(im.affine[0] * u + im.affine[1] * vv + im.affine[2]);
                    out.node.push_back(im.affine[3] * u + im.affine[4] * vv + im.affine[5]);
                }

                out.tri.push_back(remap[w[k]]);
            }

            out.label.push_back(tl[t]);
            double am;
            q.push_back(tri_quality(&P[2 * w[0]], &P[2 * w[1]], &P[2 * w[2]], &am));
            amin = std::min(amin, am);
            const double ux = P[2 * w[1]] - P[2 * w[0]], uy = P[2 * w[1] + 1] - P[2 * w[0] + 1];
            const double vx2 = P[2 * w[2]] - P[2 * w[0]], vy2 = P[2 * w[2] + 1] - P[2 * w[0] + 1];
            st.label_area[tl[t]] += 0.5 * std::fabs(ux * vy2 - uy * vx2);

            for (int k = 0; k < 3; ++k) {
                const int u2 = adj[3 * t + k];
                const int lb = u2 >= 0 ? tl[u2] : 0;

                if (lb != tl[t] && lb < tl[t]) {   // (once, from the larger label; inner on the left)
                    out.edge.insert(out.edge.end(), { w[(k + 1) % 3], w[(k + 2) % 3], tl[t], lb });
                }
            }
        }

        for (size_t e = 0; e < out.edge.size(); e += 4) {
            out.edge[e] = remap[out.edge[e]];
            out.edge[e + 1] = remap[out.edge[e + 1]];
        }

        st.nodes = out.node.size() / 2;
        st.tris = out.label.size();
        st.min_angle = st.tris ? amin : 0;

        if (!q.empty()) {
            std::vector<double> s(q);
            std::sort(s.begin(), s.end());
            st.q_min = s.front();
            st.q_p5 = s[s.size() / 20];
            st.q_median = s[s.size() / 2];
        }

        st.label_pixels.assign(F.nlab, 0.0);

        for (uint16_t l : im.lab) {
            st.label_pixels[l] += F.vx * F.vy;
        }
    }

    st.ms_mesh = ms_since(t2);
    st.ms_total = ms_since(t0);
}

}  // namespace tn
