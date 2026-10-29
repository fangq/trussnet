/***************************************************************************//**
**  \file    v2m_inflate.cpp
**  \brief   Inflated cortical / white-matter surfaces, with the mesh around them moved along
**
**  See v2m_inflate.h.
*******************************************************************************/

#include "v2m_inflate.h"
#include "v2m_omp.h"   // OmpThreadCap

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tn {

namespace {

typedef std::chrono::steady_clock clk;

struct Face {   // a triangle: sorted key and oriented corners
    int32_t k[3];
    int32_t v[3];
    bool operator<(const Face& o) const {
        return k[0] != o.k[0] ? k[0] < o.k[0] : k[1] != o.k[1] ? k[1] < o.k[1] : k[2] < o.k[2];
    }
    bool same(const Face& o) const {
        return k[0] == o.k[0] && k[1] == o.k[1] && k[2] == o.k[2];
    }
};

Face make_face(int32_t a, int32_t b, int32_t c) {
    Face f;
    f.v[0] = a;
    f.v[1] = b;
    f.v[2] = c;
    f.k[0] = a;
    f.k[1] = b;
    f.k[2] = c;
    std::sort(f.k, f.k + 3);
    return f;
}

// neighbour lists (CSR) of n nodes from elements of k nodes each, every pair an edge
struct Csr {
    std::vector<int64_t> off;
    std::vector<int32_t> nb;
};

Csr build_csr(size_t n, const std::vector<int32_t>& el, int k) {
    const size_t ne = el.size() / static_cast<size_t>(k);
    std::vector<int64_t> start(n + 1, 0);

    for (size_t e = 0; e < ne; ++e)
        for (int a = 0; a < k; ++a) {
            start[static_cast<size_t>(el[e * k + a]) + 1] += k - 1;
        }

    for (size_t i = 0; i < n; ++i) {
        start[i + 1] += start[i];
    }

    std::vector<int32_t> buf(static_cast<size_t>(start[n]));
    std::vector<int64_t> cur(start.begin(), start.end() - 1);

    for (size_t e = 0; e < ne; ++e)
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                if (a != b) {
                    buf[static_cast<size_t>(cur[static_cast<size_t>(el[e * k + a])]++)] = el[e * k + b];
                }

    std::vector<int64_t> len(n, 0);
    #pragma omp parallel for schedule(dynamic, 4096)

    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        int32_t* p = buf.data() + start[i];
        int32_t* q = buf.data() + start[i + 1];
        std::sort(p, q);
        len[i] = std::unique(p, q) - p;
    }

    Csr c;
    c.off.assign(n + 1, 0);

    for (size_t i = 0; i < n; ++i) {
        c.off[i + 1] = c.off[i] + len[i];
    }

    c.nb.resize(static_cast<size_t>(c.off[n]));
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        std::copy(buf.data() + start[i], buf.data() + start[i] + len[i], c.nb.data() + c.off[i]);
    }

    return c;
}

const double* P3(const std::vector<double>& x, int32_t v) {
    return x.data() + 3 * static_cast<size_t>(v);
}

double tet_volume(const std::vector<double>& x, const int32_t* t) {
    const double* a = P3(x, t[0]);
    const double* b = P3(x, t[1]);
    const double* c = P3(x, t[2]);
    const double* d = P3(x, t[3]);
    const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    const double v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const double w[3] = { d[0] - a[0], d[1] - a[1], d[2] - a[2] };
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])) /
           6.0;
}

const int kFaceOf[4][3] = { { 1, 2, 3 }, { 0, 3, 2 }, { 0, 1, 3 }, { 0, 2, 1 } };   // a tet's face opposite corner k

// every tet's face-neighbours (4 per tet, corner k's opposite face; -1: the mesh's boundary),
// in parallel through the nodes' tets (a global sort of the ~4 nt faces took seconds)
std::vector<int32_t> tet_adjacency(const Mesh& m) {
    const size_t nt = m.tets.size() / 4, n = m.nodes.size() / 3;
    std::vector<int64_t> off(n + 1, 0);

    for (size_t i = 0; i < 4 * nt; ++i) {
        ++off[static_cast<size_t>(m.tets[i]) + 1];
    }

    for (size_t v = 0; v < n; ++v) {
        off[v + 1] += off[v];
    }

    std::vector<int32_t> n2t(4 * nt);
    {
        std::vector<int64_t> cur(off.begin(), off.end() - 1);

        for (size_t i = 0; i < 4 * nt; ++i) {
            n2t[static_cast<size_t>(cur[static_cast<size_t>(m.tets[i])]++)] = static_cast<int32_t>(i / 4);
        }
    }
    std::vector<int32_t> nb(4 * nt, -1);
    #pragma omp parallel for schedule(static)

    for (int64_t t = 0; t < static_cast<int64_t>(nt); ++t) {
        const int32_t* v = &m.tets[4 * static_cast<size_t>(t)];

        for (int k = 0; k < 4; ++k) {
            const int32_t a = v[kFaceOf[k][0]], b = v[kFaceOf[k][1]], c = v[kFaceOf[k][2]];

            for (int64_t e = off[static_cast<size_t>(a)]; e < off[static_cast<size_t>(a) + 1]; ++e) {
                const int32_t u = n2t[static_cast<size_t>(e)];

                if (u == static_cast<int32_t>(t)) {
                    continue;
                }

                const int32_t* w = &m.tets[4 * static_cast<size_t>(u)];
                const bool hb = w[0] == b || w[1] == b || w[2] == b || w[3] == b;
                const bool hc = w[0] == c || w[1] == c || w[2] == c || w[3] == c;

                if (hb && hc) {
                    nb[4 * static_cast<size_t>(t) + static_cast<size_t>(k)] = u;
                    break;
                }
            }
        }
    }

    return nb;
}

// the kept tets' faces (outward) whose neighbour is not kept: their union's boundary
std::vector<Face> tet_boundary(const Mesh& m, const std::vector<char>& keep, const std::vector<int32_t>& nb) {
    const size_t nt = m.tets.size() / 4;
    std::vector<Face> b;

    for (size_t t = 0; t < nt; ++t) {
        if (!keep[t]) {
            continue;
        }

        const int32_t* v = &m.tets[4 * t];
        bool pos = true, got = false;   // (face corners outward for a positive tet)

        for (int k = 0; k < 4; ++k) {
            const int32_t u = nb[4 * t + static_cast<size_t>(k)];

            if (u >= 0 && keep[static_cast<size_t>(u)]) {
                continue;
            }

            if (!got) {
                pos = tet_volume(m.nodes, v) > 0.0;
                got = true;
            }

            b.push_back(pos ? make_face(v[kFaceOf[k][0]], v[kFaceOf[k][1]], v[kFaceOf[k][2]])
                        : make_face(v[kFaceOf[k][0]], v[kFaceOf[k][2]], v[kFaceOf[k][1]]));
        }
    }

    return b;
}

// The plane between two halves (the hemispheres' midline): x + a y + b z = c, the one across
// which the least of the region lies -- the points' weights (tets' volumes, faces' areas) in a
// 1 mm slab, 5 mm smoothed -- with |a|, |b| <= 0.3 (~17 degrees of yaw and roll) and c within
// 20 mm of the weighted centre: through the corpus callosum and the brainstem, or the
// fissure's gap, however the head is turned. A coarse search (0.05), then a fine one (0.01).
std::array<double, 3> mid_plane(const std::vector<double>& pt, const std::vector<double>& w) {
    const size_t np = w.size();
    const size_t stride = std::max<size_t>(1, np / 1000000);   // (a million points at most)
    const int nbin = 41;
    auto cost = [&](double a, double b, double& c) {
        double sw = 0.0, sd = 0.0;

        for (size_t i = 0; i < np; i += stride) {
            sw += w[i];
            sd += w[i] * (pt[3 * i] + a * pt[3 * i + 1] + b * pt[3 * i + 2]);
        }

        const double lo = (sw > 0.0 ? sd / sw : 0.0) - 0.5 * nbin;
        std::array<double, nbin> h {};

        for (size_t i = 0; i < np; i += stride) {
            const double d = pt[3 * i] + a * pt[3 * i + 1] + b * pt[3 * i + 2] - lo;

            if (d >= 0.0 && d < nbin) {
                h[static_cast<size_t>(d)] += w[i];
            }
        }

        // (a slab 1 / |n| thick in d: its weight x |n| is the cross-section's)
        const double nn = std::sqrt(1.0 + a * a + b * b);
        double best = 1e300;

        for (int k = 2; k < nbin - 2; ++k) {
            const double v = (h[k - 2] + h[k - 1] + h[k] + h[k + 1] + h[k + 2]) * nn
                             + 1e-9 * std::abs(k - nbin / 2);   // (ties: the most central)

            if (v < best) {
                best = v;
                c = lo + k + 0.5;
            }
        }

        return best;
    };
    std::array<double, 3> pl = { 0.0, 0.0, 0.0 };
    double cbest = 1e300;
    const double steps[2] = { 0.05, 0.01 };
    const int half[2] = { 6, 4 };

    for (int lv = 0; lv < 2; ++lv) {
        const double a0 = pl[0], b0 = pl[1];
        const int nk = 2 * half[lv] + 1;
        std::vector<double> cs(static_cast<size_t>(nk * nk)), cc(static_cast<size_t>(nk * nk));
        #pragma omp parallel for schedule(dynamic)

        for (int q = 0; q < nk * nk; ++q) {
            const double a = a0 + steps[lv] * (q / nk - half[lv]), b = b0 + steps[lv] * (q % nk - half[lv]);
            cs[static_cast<size_t>(q)] = cost(a, b, cc[static_cast<size_t>(q)]);
        }

        for (int q = 0; q < nk * nk; ++q)
            if (cs[static_cast<size_t>(q)] < cbest) {
                cbest = cs[static_cast<size_t>(q)];
                pl = { a0 + steps[lv] * (q / nk - half[lv]), b0 + steps[lv] * (q % nk - half[lv]), cc[static_cast<size_t>(q)] };
            }
    }

    return pl;
}

// the tets of a set's region: its labels, its half, minus its exclusions' parts, then cleaned
// (a tet attached to it by at most one face -- a spike's tip, a fin -- dropped; an outside
// tet touching it on three or more faces -- a notch -- added; a few passes): one-tet
// protrusions would end as needles of the smoothed surface
std::vector<char> region_tets(const Mesh& m, const InflateSet& s, const std::vector<int32_t>& nb, size_t& excluded,
                              size_t& cleaned, std::array<double, 3>& plane) {
    const size_t nt = m.tets.size() / 4;
    int maxlab = 0;

    for (int l : s.labels) {
        maxlab = std::max(maxlab, l);
    }

    for (int l : s.exclude) {
        maxlab = std::max(maxlab, l);
    }

    const int seed = s.seed >= 0 ? s.seed : (s.labels.empty() ? -1 : s.labels[0]);
    maxlab = std::max(maxlab, seed);
    std::vector<char> in(static_cast<size_t>(maxlab) + 1, 0), ex(static_cast<size_t>(maxlab) + 1, 0);

    for (int l : s.labels) {
        in[static_cast<size_t>(l)] = 1;
    }

    for (int l : s.exclude) {
        ex[static_cast<size_t>(l)] = 1;
    }

    auto lab = [&](size_t t) {
        return m.tet_labels[t];
    };
    auto inl = [&](const std::vector<char>& tab, size_t t) {
        return lab(t) >= 0 && lab(t) <= maxlab && tab[static_cast<size_t>(lab(t))];
    };
    std::vector<char> keep(nt);

    for (size_t t = 0; t < nt; ++t) {
        keep[t] = inl(in, t);
    }

    auto centre = [&](size_t t, double* c) {
        c[0] = c[1] = c[2] = 0.0;

        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 3; ++j) {
                c[j] += 0.25 * m.nodes[3 * static_cast<size_t>(m.tets[4 * t + k]) + j];
            }
    };

    if (s.side != 0) {   // one half: the tets on that side of the region's midline plane
        std::vector<double> pt, w;

        for (size_t t = 0; t < nt; ++t)
            if (keep[t]) {
                double c[3];
                centre(t, c);
                pt.insert(pt.end(), c, c + 3);
                const int32_t* v = &m.tets[4 * t];
                w.push_back(std::abs(tet_volume(m.nodes, v)));
            }

        plane = mid_plane(pt, w);
        #pragma omp parallel for schedule(static)

        for (int64_t t = 0; t < static_cast<int64_t>(nt); ++t)
            if (keep[static_cast<size_t>(t)]) {
                double c[3];
                centre(static_cast<size_t>(t), c);
                const bool left = c[0] + plane[0] * c[1] + plane[1] * c[2] < plane[2];
                keep[static_cast<size_t>(t)] = s.side < 0 ? left : !left;
            }
    }

    if (!s.exclude.empty() && seed >= 0) {
        // grown through the region, the excluded tissue and the seed's, from the seed label (0)
        // and from the excluded labels (1), level by level: the first to reach a tet owns it
        std::vector<char> grow(nt);

        for (size_t t = 0; t < nt; ++t) {
            grow[t] = keep[t] || inl(ex, t) || lab(t) == seed;
        }

        std::vector<int8_t> owner(nt, -1);
        std::vector<int32_t> front;

        for (size_t t = 0; t < nt; ++t)
            if (lab(t) == seed) {
                owner[t] = 0;
                front.push_back(static_cast<int32_t>(t));
            } else if (grow[t] && !keep[t]) {
                owner[t] = 1;
                front.push_back(static_cast<int32_t>(t));
            }

        while (!front.empty()) {
            std::vector<int32_t> next;

            for (int32_t t : front)
                for (int k = 0; k < 4; ++k) {
                    const int32_t u = nb[4 * static_cast<size_t>(t) + static_cast<size_t>(k)];

                    if (u >= 0 && grow[static_cast<size_t>(u)] && owner[static_cast<size_t>(u)] < 0) {
                        owner[static_cast<size_t>(u)] = owner[static_cast<size_t>(t)];
                        next.push_back(u);
                    }
                }

            front.swap(next);
        }

        for (size_t t = 0; t < nt; ++t)
            if (keep[t] && owner[t] != 0) {
                keep[t] = 0;
                ++excluded;
            }
    }

    for (int pass = 0; pass < 5; ++pass) {
        std::vector<int32_t> flip;

        for (size_t t = 0; t < nt; ++t) {
            int in_ = 0;

            for (int k = 0; k < 4; ++k) {
                const int32_t u = nb[4 * t + static_cast<size_t>(k)];
                in_ += u >= 0 && keep[static_cast<size_t>(u)];
            }

            if ((keep[t] && in_ <= 1) || (!keep[t] && in_ >= 3)) {
                flip.push_back(static_cast<int32_t>(t));
            }
        }

        if (flip.empty()) {
            break;
        }

        for (int32_t t : flip) {
            keep[static_cast<size_t>(t)] = !keep[static_cast<size_t>(t)];
        }

        cleaned += flip.size();
    }

    return keep;
}

// Jacobi-preconditioned CG on three right-hand sides at once (interleaved x y z), A applied by
// `apply`, x warm-started; each component stopped when its preconditioned residual is below
// tol^2 of b's (a relative 1e-6: 1e-4 mm on a head) -- the vector updates and the dot
// products in parallel
int pcg3(size_t m, const std::vector<double>& diag,
          const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply, const std::vector<double>& b,
          std::vector<double>& x, double tol, int maxit) {
    std::vector<double> r(3 * m), z(3 * m), q(3 * m), Aq(3 * m);
    apply(x, Aq);
    double rz0 = 0, rz1 = 0, rz2 = 0, b0 = 0, b1 = 0, b2 = 0;
    #pragma omp parallel for reduction(+ : rz0, rz1, rz2, b0, b1, b2) schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
        const size_t j = 3 * static_cast<size_t>(i);

        for (int c = 0; c < 3; ++c) {
            r[j + c] = b[j + c] - Aq[j + c];
            z[j + c] = r[j + c] / diag[static_cast<size_t>(i)];
            q[j + c] = z[j + c];
        }

        rz0 += r[j] * z[j];
        rz1 += r[j + 1] * z[j + 1];
        rz2 += r[j + 2] * z[j + 2];
        b0 += b[j] * b[j] / diag[static_cast<size_t>(i)];
        b1 += b[j + 1] * b[j + 1] / diag[static_cast<size_t>(i)];
        b2 += b[j + 2] * b[j + 2] / diag[static_cast<size_t>(i)];
    }

    double rz[3] = { rz0, rz1, rz2 };
    const double bb[3] = { std::max(b0, 1e-30), std::max(b1, 1e-30), std::max(b2, 1e-30) };

    int it = 0;

    for (; it < maxit; ++it) {
        if (rz[0] <= tol * tol * bb[0] && rz[1] <= tol * tol * bb[1] && rz[2] <= tol * tol * bb[2]) {
            break;
        }

        apply(q, Aq);
        double qa0 = 0, qa1 = 0, qa2 = 0;
        #pragma omp parallel for reduction(+ : qa0, qa1, qa2) schedule(static)

        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
            const size_t j = 3 * static_cast<size_t>(i);
            qa0 += q[j] * Aq[j];
            qa1 += q[j + 1] * Aq[j + 1];
            qa2 += q[j + 2] * Aq[j + 2];
        }

        const double al[3] = { qa0 > 0.0 ? rz[0] / qa0 : 0.0, qa1 > 0.0 ? rz[1] / qa1 : 0.0, qa2 > 0.0 ? rz[2] / qa2 : 0.0 };
        double n0 = 0, n1 = 0, n2 = 0;
        #pragma omp parallel for reduction(+ : n0, n1, n2) schedule(static)

        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
            const size_t j = 3 * static_cast<size_t>(i);

            for (int c = 0; c < 3; ++c) {
                x[j + c] += al[c] * q[j + c];
                r[j + c] -= al[c] * Aq[j + c];
                z[j + c] = r[j + c] / diag[static_cast<size_t>(i)];
            }

            n0 += r[j] * z[j];
            n1 += r[j + 1] * z[j + 1];
            n2 += r[j + 2] * z[j + 2];
        }

        const double be[3] = { rz[0] > 0.0 ? n0 / rz[0] : 0.0, rz[1] > 0.0 ? n1 / rz[1] : 0.0, rz[2] > 0.0 ? n2 / rz[2] : 0.0 };
        #pragma omp parallel for schedule(static)

        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i)
            for (int c = 0; c < 3; ++c) {
                q[3 * static_cast<size_t>(i) + c] = z[3 * static_cast<size_t>(i) + c] + be[c] * q[3 * static_cast<size_t>(i) + c];
            }

        rz[0] = n0;
        rz[1] = n1;
        rz[2] = n2;
    }

    return it;
}

// The pinches split: a vertex whose faces form several fans (sheets meeting only at the
// vertex, or along an edge of 4+ faces) becomes one local vertex per fan. Returns the
// locals' nodes and the faces in locals.
struct Surf {
    std::vector<int32_t> node;
    std::vector<int32_t> tri;
};

Surf split_pinches(const std::vector<Face>& F, size_t n, size_t& nsplit) {
    const size_t nf = F.size();
    // each face-edge's partner face (the edge's other face when it has exactly two)
    std::vector<std::pair<int64_t, int32_t>> ek;
    ek.reserve(3 * nf);

    for (size_t f = 0; f < nf; ++f)
        for (int a = 0; a < 3; ++a) {
            const int64_t i = F[f].v[a], j = F[f].v[(a + 1) % 3];
            ek.push_back({ std::min(i, j) * static_cast<int64_t>(n) + std::max(i, j), static_cast<int32_t>(3 * f + a) });
        }

    std::sort(ek.begin(), ek.end());
    std::vector<int32_t> partner(3 * nf, -1);

    for (size_t i = 0; i < ek.size();) {
        size_t j = i + 1;

        while (j < ek.size() && ek[j].first == ek[i].first) {
            ++j;
        }

        if (j - i == 2) {
            partner[static_cast<size_t>(ek[i].second)] = ek[i + 1].second / 3;
            partner[static_cast<size_t>(ek[i + 1].second)] = ek[i].second / 3;
        }

        i = j;
    }

    // vertex -> its faces
    std::vector<int64_t> off(n + 1, 0);

    for (size_t f = 0; f < nf; ++f)
        for (int a = 0; a < 3; ++a) {
            ++off[static_cast<size_t>(F[f].v[a]) + 1];
        }

    for (size_t v = 0; v < n; ++v) {
        off[v + 1] += off[v];
    }

    std::vector<int32_t> vf(static_cast<size_t>(off[n]));
    std::vector<int64_t> cur(off.begin(), off.end() - 1);

    for (size_t f = 0; f < nf; ++f)
        for (int a = 0; a < 3; ++a) {
            vf[static_cast<size_t>(cur[static_cast<size_t>(F[f].v[a])]++)] = static_cast<int32_t>(f);
        }

    Surf s;
    s.tri.assign(3 * nf, -1);
    std::vector<int32_t> star, dsu;
    auto find = [&](int32_t i) {
        while (dsu[static_cast<size_t>(i)] != i) {
            dsu[static_cast<size_t>(i)] = dsu[static_cast<size_t>(dsu[static_cast<size_t>(i)])];
            i = dsu[static_cast<size_t>(i)];
        }

        return i;
    };

    for (size_t v = 0; v < n; ++v) {
        if (off[v + 1] == off[v]) {
            continue;
        }

        star.assign(vf.begin() + off[v], vf.begin() + off[v + 1]);
        dsu.resize(star.size());
        std::iota(dsu.begin(), dsu.end(), 0);

        for (size_t i = 0; i < star.size(); ++i) {
            const size_t f = static_cast<size_t>(star[i]);
            int c = 0;

            while (F[f].v[c] != static_cast<int32_t>(v)) {
                ++c;
            }

            // the face's two edges at v: corners (c, c+1) and (c-1, c)
            for (int a : { c, (c + 2) % 3 }) {
                const int32_t g = partner[3 * f + static_cast<size_t>(a)];

                if (g < 0) {
                    continue;
                }

                for (size_t j = 0; j < star.size(); ++j)
                    if (star[j] == g) {
                        dsu[static_cast<size_t>(find(static_cast<int32_t>(i)))] = find(static_cast<int32_t>(j));
                        break;
                    }
            }
        }

        std::vector<std::pair<int32_t, int32_t>> fans;   // (root, local)

        for (size_t i = 0; i < star.size(); ++i) {
            const int32_t r = find(static_cast<int32_t>(i));
            int32_t loc = -1;

            for (const auto& fr : fans)
                if (fr.first == r) {
                    loc = fr.second;
                }

            if (loc < 0) {
                loc = static_cast<int32_t>(s.node.size());
                s.node.push_back(static_cast<int32_t>(v));
                fans.push_back({ r, loc });
            }

            const size_t f = static_cast<size_t>(star[i]);

            for (int c = 0; c < 3; ++c)
                if (F[f].v[c] == static_cast<int32_t>(v)) {
                    s.tri[3 * f + static_cast<size_t>(c)] = loc;
                }
        }

        nsplit += fans.size() - 1;
    }

    return s;
}

}  // namespace

std::vector<InflateSet> parse_inflate_sets(const std::string& spec) {
    // the SIAM presets: "white" (the cerebral WM and the deep structures it encloses, the
    // cerebellar WM's part dropped, grown from the cortex) and "pial" / "gm" (the same with the
    // cortex), each as two hemispheres
    std::string s;
    size_t r0 = 0;

    while (r0 <= spec.size()) {
        const size_t r1 = std::min(spec.find(',', r0), spec.size());
        const std::string item = spec.substr(r0, r1 - r0);
        r0 = r1 + 1;
        const char* set = item == "white" ? "2+4+6+7+8+9+10+11+12" : item == "pial" || item == "gm" ? "1+2+4+6+7+8+9+10+11+12" : nullptr;
        const char* tail = item == "white" ? "-5^1" : "-5";
        const std::string nm = item == "gm" ? "pial" : item;
        s += s.empty() ? "" : ",";
        s += set ? "lh_" + nm + "=" + set + "@L" + tail + ",rh_" + nm + "=" + set + "@R" + tail : item;
    }

    std::vector<InflateSet> sets;
    size_t p0 = 0;

    while (p0 <= s.size()) {
        const size_t p1 = std::min(s.find(',', p0), s.size());
        std::string item = s.substr(p0, p1 - p0);
        p0 = p1 + 1;

        if (item.empty()) {
            continue;
        }

        InflateSet set;
        set.name = "surface" + std::to_string(sets.size() + 1);
        const size_t eq = item.find('=');

        if (eq != std::string::npos) {
            set.name = item.substr(0, eq);
            item = item.substr(eq + 1);

            if (set.name.empty() || set.name.find_first_of("()\"") != std::string::npos) {
                throw std::runtime_error("inflate: bad surface name in '" + s + "'");
            }
        }

        auto label = [&](const std::string& tok) {
            char* end = nullptr;
            const long v = std::strtol(tok.c_str(), &end, 10);

            if (tok.empty() || *end != '\0' || v < 0) {
                throw std::runtime_error("inflate: bad label '" + tok + "' in '" + s +
                                         "' (want [name=]L1+L2..[@L|@R][-E..], sets joined by ,)");
            }

            return static_cast<int>(v);
        };
        // "^L" the exclusions' seed, "-E" the exclusions (after the labels and the side)
        const size_t car = item.find('^');

        if (car != std::string::npos) {
            const size_t ce = item.find_first_of("-@", car + 1);
            set.seed = label(item.substr(car + 1, (ce == std::string::npos ? item.size() : ce) - car - 1));
            item = item.substr(0, car) + (ce == std::string::npos ? "" : item.substr(ce));
        }

        size_t dash = item.find('-');

        while (dash != std::string::npos) {
            const size_t nx = item.find('-', dash + 1);
            set.exclude.push_back(label(item.substr(dash + 1, (nx == std::string::npos ? item.size() : nx) - dash - 1)));
            item = item.substr(0, dash) + (nx == std::string::npos ? "" : item.substr(nx));
            dash = item.find('-');
        }

        const size_t at = item.find('@');

        if (at != std::string::npos) {
            const std::string h = item.substr(at + 1);
            set.side = h == "L" || h == "l" ? -1 : h == "R" || h == "r" ? 1 : 2;

            if (set.side == 2) {
                throw std::runtime_error("inflate: bad side '@" + h + "' in '" + s + "' (want @L or @R)");
            }

            item = item.substr(0, at);
        }

        size_t q0 = 0;

        while (q0 <= item.size()) {
            const size_t q1 = std::min(item.find('+', q0), item.size());
            set.labels.push_back(label(item.substr(q0, q1 - q0)));
            q0 = q1 + 1;
        }

        sets.push_back(set);
    }

    if (sets.empty()) {
        throw std::runtime_error("inflate: no label set in '" + s + "'");
    }

    return sets;
}

// (nbp: the tets' adjacency when already built -- one build for every set)
static void inflate_impl(Mesh& m, const InflateParams& p, InflateStats& st, const std::vector<int32_t>* nbp) {
    const clk::time_point t0 = clk::now();
    st = InflateStats();
    const size_t n = m.nodes.size() / 3, nt = m.tets.size() / 4, ntri = m.tris.size() / 3;
    const bool tets = nt > 0 && m.tet_labels.size() == nt;
    const bool move = tets && p.move_mesh;   // (the tets follow the surfaces)

    if (!tets && (ntri == 0 || m.tri_labels.size() != 2 * ntri)) {
        throw std::runtime_error("inflate: the mesh has neither labelled tets nor labelled triangles (MeshTri inner outer)");
    }

    std::vector<int32_t> nb_own;

    if (tets && !nbp) {
        nb_own = tet_adjacency(m);
        nbp = &nb_own;
    }

    // surfaces only, several sets: each inflated on its own (sets sharing faces -- two halves'
    // cut -- would otherwise join into one surface, pinched along the cut)
    if (!p.move_mesh && p.sets.size() > 1) {
        std::vector<Mesh::NamedSurface> all;
        st.sulc_min = 1e30;
        st.sulc_max = -1e30;

        for (const InflateSet& set : p.sets) {
            InflateParams q = p;
            q.sets = { set };
            InflateStats sk;
            inflate_impl(m, q, sk, nbp);   // (the mesh itself: the surfaces alone change)
            for (Mesh::NamedSurface& o : m.surfaces) {   // (the set's surface, and its original)
                all.push_back(std::move(o));
            }

            st.surf_nodes += sk.surf_nodes;
            st.surf_faces += sk.surf_faces;
            st.components += sk.components;
            st.split += sk.split;
            st.excluded += sk.excluded;
            st.cleaned += sk.cleaned;
            st.folds += sk.folds;
            st.closed += sk.closed;
            st.area += sk.area;
            st.volume += sk.volume;
            st.iters_done = sk.iters_done;
            st.by_area = sk.by_area;
            st.sulc_min = std::min(st.sulc_min, sk.sulc_min);
            st.sulc_max = std::max(st.sulc_max, sk.sulc_max);
        }

        m.surfaces = std::move(all);
        st.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        return;
    }

    const bool by_area = p.keep == 1 || (p.keep == 0 && !move);
    st.by_area = by_area;
    clk::time_point tph = clk::now();
    std::vector<double> tacc(8, 0.0);   // per-step phases: smooth, relax, unfold, rescale, mesh
    auto phase = [&](const char* what) {   // (-v: the setup's phases as they end)
        if (p.verbose) {
            std::fprintf(stderr, "[infl]  %-22s %8.0f ms\n", what, std::chrono::duration<double, std::milli>(clk::now() - tph).count());
        }

        tph = clk::now();
    };
    auto lap = [&](int k) {
        tacc[static_cast<size_t>(k)] += std::chrono::duration<double, std::milli>(clk::now() - tph).count();
        tph = clk::now();
    };

    // 1. each set's surface: its region's boundary, outward, its pinches split
    std::vector<int32_t> sv, sf;              // the locals' nodes; the faces, in locals
    std::vector<size_t> l0(1, 0), f0(1, 0);   // each set's first local / face

    for (const InflateSet& set : p.sets) {
        std::vector<Face> F;
        std::array<double, 3> plane = { 0.0, 0.0, 0.0 };

        if (tets) {
            F = tet_boundary(m, region_tets(m, set, *nbp, st.excluded, st.cleaned, plane), *nbp);
        } else {   // MeshTri: inner -> outer is the normal's direction
            int maxlab = 0;

            for (int l : set.labels) {
                maxlab = std::max(maxlab, l);
            }

            std::vector<char> in(static_cast<size_t>(maxlab) + 1, 0);

            for (int l : set.labels) {
                in[static_cast<size_t>(l)] = 1;
            }

            auto inset = [&](int l) {
                return l >= 0 && l <= maxlab && in[static_cast<size_t>(l)];
            };

            for (size_t f = 0; f < ntri; ++f) {
                const bool a = inset(m.tri_labels[2 * f]), b = inset(m.tri_labels[2 * f + 1]);

                if (a != b) {
                    const int32_t* v = &m.tris[3 * f];
                    F.push_back(a ? make_face(v[0], v[1], v[2]) : make_face(v[0], v[2], v[1]));
                }
            }

            if (set.side != 0 && !F.empty()) {   // one half, by the midline plane (its cut stays open)
                std::vector<double> pt, w;

                for (const Face& f : F) {
                    const double* a = &m.nodes[3 * static_cast<size_t>(f.v[0])];
                    const double* b = &m.nodes[3 * static_cast<size_t>(f.v[1])];
                    const double* c = &m.nodes[3 * static_cast<size_t>(f.v[2])];
                    const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, t[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
                    const double cr[3] = { u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0] };
                    pt.insert(pt.end(), { (a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0 });
                    w.push_back(0.5 * std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]));
                }

                plane = mid_plane(pt, w);
                std::vector<Face> half;

                for (size_t i = 0; i < F.size(); ++i) {
                    const bool left = pt[3 * i] + plane[0] * pt[3 * i + 1] + plane[1] * pt[3 * i + 2] < plane[2];

                    if (set.side < 0 ? left : !left) {
                        half.push_back(F[i]);
                    }
                }

                F.swap(half);
            }
        }

        if (p.verbose && set.side != 0) {
            const double deg = 180.0 / 3.14159265358979323846;
            std::fprintf(stderr, "[infl]  %s: midline x + %.2f y + %.2f z = %.1f (yaw %.1f, roll %.1f deg)\n", set.name.c_str(),
                         plane[0], plane[1], plane[2], std::atan(plane[0]) * deg, std::atan(plane[1]) * deg);
        }

        std::sort(F.begin(), F.end());
        F.erase(std::unique(F.begin(), F.end(), [](const Face& a, const Face& b) {
            return a.same(b);
        }), F.end());
        const Surf s = split_pinches(F, n, st.split);
        const int32_t base = static_cast<int32_t>(sv.size());
        sv.insert(sv.end(), s.node.begin(), s.node.end());

        for (int32_t v : s.tri) {
            sf.push_back(base + v);
        }

        l0.push_back(sv.size());
        f0.push_back(sf.size() / 3);
    }

    if (sf.empty()) {
        throw std::runtime_error("inflate: no surface: none of the label sets has a boundary in the mesh");
    }

    phase("regions + surfaces");

    const size_t ns = sv.size(), nf = sf.size() / 3;
    st.surf_nodes = ns;
    st.surf_faces = nf;
    const Csr sg = build_csr(ns, sf, 3);
    std::vector<double>& x = m.nodes;
    std::vector<double> y(3 * ns);   // the surfaces' positions (a split vertex's copies apart)

    for (size_t k = 0; k < ns; ++k)
        for (int c = 0; c < 3; ++c) {
            y[3 * k + c] = x[3 * static_cast<size_t>(sv[k]) + c];
        }

    std::vector<char> issurf(n, 0);

    for (int32_t v : sv) {
        issurf[static_cast<size_t>(v)] = 1;
    }

    // the moving mesh's outer boundary: fixed (an inflated vertex on it too)
    std::vector<char> fixed(n, 0);

    if (move) {
        for (const Face& f : tet_boundary(m, std::vector<char>(nt, 1), *nbp))
            for (int a = 0; a < 3; ++a) {
                fixed[static_cast<size_t>(f.v[a])] = 1;
            }
    }

    auto held = [&](size_t k) {
        return fixed[static_cast<size_t>(sv[k])] != 0;
    };

    // the surface pre-smoothed (FreeSurfer's smoothwm): Taubin passes (lambda 0.5 / mu -0.53, no
    // shrinking) -- the voxel staircase out; the inflation, sulc and curv start from it
    {
        std::vector<double> t(3 * ns);

        for (int pass = 0; pass < 2 * p.presmooth; ++pass) {
            const double w = pass % 2 == 0 ? 0.5 : -0.53;
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                const int64_t deg = sg.off[k + 1] - sg.off[k];
                double a[3] = { 0, 0, 0 };

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e)
                    for (int c = 0; c < 3; ++c) {
                        a[c] += y[3 * static_cast<size_t>(sg.nb[e]) + c];
                    }

                for (int c = 0; c < 3; ++c) {
                    t[3 * static_cast<size_t>(k) + c] = held(static_cast<size_t>(k)) || deg == 0 ? y[3 * static_cast<size_t>(k) + c]
                                                        : y[3 * static_cast<size_t>(k) + c] +
                                                        w * (a[c] / static_cast<double>(deg) - y[3 * static_cast<size_t>(k) + c]);
                }
            }

            y.swap(t);
        }
    }
    const std::vector<double> y0 = y;

    // connected surfaces, each closed when every edge has an even number of its faces
    std::vector<int32_t> comp(ns, -1);
    int32_t nc = 0;

    for (size_t s = 0; s < ns; ++s) {
        if (comp[s] >= 0) {
            continue;
        }

        std::vector<int32_t> stack(1, static_cast<int32_t>(s));
        comp[s] = nc;

        while (!stack.empty()) {
            const int32_t u = stack.back();
            stack.pop_back();

            for (int64_t e = sg.off[u]; e < sg.off[u + 1]; ++e)
                if (comp[static_cast<size_t>(sg.nb[e])] < 0) {
                    comp[static_cast<size_t>(sg.nb[e])] = nc;
                    stack.push_back(sg.nb[e]);
                }
        }

        ++nc;
    }

    st.components = static_cast<size_t>(nc);
    const size_t ncs = static_cast<size_t>(nc);
    std::vector<char> closed(ncs, 1);
    {
        std::vector<int64_t> keys;
        keys.reserve(3 * nf);

        for (size_t f = 0; f < nf; ++f)
            for (int a = 0; a < 3; ++a) {
                const int64_t i = sf[3 * f + a], j = sf[3 * f + (a + 1) % 3];
                keys.push_back(std::min(i, j) * static_cast<int64_t>(ns) + std::max(i, j));
            }

        std::sort(keys.begin(), keys.end());

        for (size_t i = 0; i < keys.size();) {
            size_t j = i + 1;

            while (j < keys.size() && keys[j] == keys[i]) {
                ++j;
            }

            if ((j - i) % 2 != 0) {
                closed[static_cast<size_t>(comp[static_cast<size_t>(keys[i] / static_cast<int64_t>(ns))])] = 0;
            }

            i = j;
        }
    }
    auto face_area = [&](const std::vector<double>& q, size_t f) {
        const double* a = &q[3 * static_cast<size_t>(sf[3 * f])];
        const double* b = &q[3 * static_cast<size_t>(sf[3 * f + 1])];
        const double* c = &q[3 * static_cast<size_t>(sf[3 * f + 2])];
        const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const double v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const double cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
        return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    };
    auto comp_area = [&](const std::vector<double>& q, std::vector<double>& area) {
        area.assign(ncs, 0.0);

        for (size_t f = 0; f < nf; ++f) {
            area[static_cast<size_t>(comp[static_cast<size_t>(sf[3 * f])])] += face_area(q, f);
        }
    };
    auto comp_volume = [&](const std::vector<double>& q, std::vector<double>& vol) {   // the divergence theorem
        vol.assign(ncs, 0.0);

        for (size_t f = 0; f < nf; ++f) {
            const double* a = &q[3 * static_cast<size_t>(sf[3 * f])];
            const double* b = &q[3 * static_cast<size_t>(sf[3 * f + 1])];
            const double* c = &q[3 * static_cast<size_t>(sf[3 * f + 2])];
            vol[static_cast<size_t>(comp[static_cast<size_t>(sf[3 * f])])] +=
                (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
        }
    };
    // vertex normals (area-weighted, outward, unit)
    auto normals = [&](const std::vector<double>& q, std::vector<double>& nr) {
        nr.assign(3 * ns, 0.0);

        for (size_t f = 0; f < nf; ++f) {
            const double* a = &q[3 * static_cast<size_t>(sf[3 * f])];
            const double* b = &q[3 * static_cast<size_t>(sf[3 * f + 1])];
            const double* c = &q[3 * static_cast<size_t>(sf[3 * f + 2])];
            const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
            const double v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
            const double cr[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };

            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    nr[3 * static_cast<size_t>(sf[3 * f + j]) + k] += cr[k];
                }
        }

        for (size_t k = 0; k < ns; ++k) {
            const double l = std::sqrt(nr[3 * k] * nr[3 * k] + nr[3 * k + 1] * nr[3 * k + 1] + nr[3 * k + 2] * nr[3 * k + 2]);

            for (int c = 0; c < 3; ++c) {
                nr[3 * k + c] = l > 0.0 ? nr[3 * k + c] / l : 0.0;
            }
        }
    };
    // local -> its faces
    std::vector<int64_t> vfo(ns + 1, 0);

    for (size_t i = 0; i < 3 * nf; ++i) {
        ++vfo[static_cast<size_t>(sf[i]) + 1];
    }

    for (size_t k = 0; k < ns; ++k) {
        vfo[k + 1] += vfo[k];
    }

    std::vector<int32_t> vfl(3 * nf);
    {
        std::vector<int64_t> cur(vfo.begin(), vfo.end() - 1);

        for (size_t i = 0; i < 3 * nf; ++i) {
            vfl[static_cast<size_t>(cur[static_cast<size_t>(sf[i])]++)] = static_cast<int32_t>(i / 3);
        }
    }
    auto face_n = [&](const std::vector<double>& q, size_t f, const double* moved, size_t mv, double* out) {
        // (the face's cross product, its vertex mv at `moved` when given)
        const double* a = mv == static_cast<size_t>(sf[3 * f]) && moved ? moved : &q[3 * static_cast<size_t>(sf[3 * f])];
        const double* b = mv == static_cast<size_t>(sf[3 * f + 1]) && moved ? moved : &q[3 * static_cast<size_t>(sf[3 * f + 1])];
        const double* c = mv == static_cast<size_t>(sf[3 * f + 2]) && moved ? moved : &q[3 * static_cast<size_t>(sf[3 * f + 2])];
        const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const double v[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        out[0] = u[1] * v[2] - u[2] * v[1];
        out[1] = u[2] * v[0] - u[0] * v[2];
        out[2] = u[0] * v[1] - u[1] * v[0];
    };
    // vertex normals smoothed over a few rings: the reference a face is folded against
    // vertex normals smoothed over a few rings: the reference a face is folded against
    // (in parallel: each vertex gathers its faces' cross products, then its ring's normals)
    auto ref_normals = [&](const std::vector<double>& q, std::vector<double>& rn) {
        rn.assign(3 * ns, 0.0);
        #pragma omp parallel for schedule(static)

        for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
            double a[3] = { 0, 0, 0 };

            for (int64_t e = vfo[static_cast<size_t>(k)]; e < vfo[static_cast<size_t>(k) + 1]; ++e) {
                double fn[3];
                face_n(q, static_cast<size_t>(vfl[static_cast<size_t>(e)]), nullptr, 0, fn);

                for (int c = 0; c < 3; ++c) {
                    a[c] += fn[c];
                }
            }

            for (int c = 0; c < 3; ++c) {
                rn[3 * static_cast<size_t>(k) + c] = a[c];
            }
        }

        std::vector<double> t(3 * ns);

        for (int pass = 0; pass < 5; ++pass) {
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                double a[3] = { rn[3 * static_cast<size_t>(k)], rn[3 * static_cast<size_t>(k) + 1], rn[3 * static_cast<size_t>(k) + 2] };

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e)
                    for (int c = 0; c < 3; ++c) {
                        a[c] += rn[3 * static_cast<size_t>(sg.nb[e]) + c];
                    }

                const double l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

                for (int c = 0; c < 3; ++c) {
                    t[3 * static_cast<size_t>(k) + c] = l > 0.0 ? a[c] / l : 0.0;
                }
            }

            rn.swap(t);
        }
    };
    auto folded = [&](const std::vector<double>& q, const std::vector<double>& rn, size_t f) {
        double fn[3];
        face_n(q, f, nullptr, 0, fn);
        double r = 0.0;

        for (int j = 0; j < 3; ++j) {
            const size_t k = static_cast<size_t>(sf[3 * f + j]);
            r += fn[0] * rn[3 * k] + fn[1] * rn[3 * k + 1] + fn[2] * rn[3 * k + 2];
        }

        return r < 0.0;
    };
    // the folds smoothed out: the vertices of folded faces and their ring to their
    // neighbours' mean (Jacobi), until none is left, the count stalls (no 5% drop in 3 passes:
    // the rest are creases smoothing cannot open) or 30 passes; returns the faces still folded
    std::vector<double> rnrm;
    auto unfold = [&](std::vector<double>& q) {
        size_t left = 0;
        std::vector<char> fo(nf), mk(ns);
        std::vector<double> t(3 * ns);
        size_t best = SIZE_MAX;
        int stall = 0, pass = 0;

        for (; pass < 30; ++pass) {
            ref_normals(q, rnrm);
            size_t cnt = 0;
            #pragma omp parallel for reduction(+ : cnt) schedule(static)

            for (int64_t f = 0; f < static_cast<int64_t>(nf); ++f) {
                fo[static_cast<size_t>(f)] = folded(q, rnrm, static_cast<size_t>(f));
                cnt += fo[static_cast<size_t>(f)];
            }

            left = cnt;

            if (cnt < best - best / 20) {
                best = cnt;
                stall = 0;
            } else if (++stall >= 3) {
                break;
            }

            if (left == 0) {
                break;
            }

            // (a vertex is marked when it or a neighbour has a folded face)
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                char hit = 0;

                for (int64_t e = vfo[static_cast<size_t>(k)]; e < vfo[static_cast<size_t>(k) + 1] && !hit; ++e) {
                    hit = fo[static_cast<size_t>(vfl[static_cast<size_t>(e)])];
                }

                for (int64_t e = sg.off[k]; e < sg.off[k + 1] && !hit; ++e) {
                    const size_t u = static_cast<size_t>(sg.nb[e]);

                    for (int64_t g = vfo[u]; g < vfo[u + 1] && !hit; ++g) {
                        hit = fo[static_cast<size_t>(vfl[static_cast<size_t>(g)])];
                    }
                }

                mk[static_cast<size_t>(k)] = hit;
            }

            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                const int64_t deg = sg.off[k + 1] - sg.off[k];

                for (int c = 0; c < 3; ++c) {
                    t[3 * static_cast<size_t>(k) + c] = q[3 * static_cast<size_t>(k) + c];
                }

                if (!mk[static_cast<size_t>(k)] || held(static_cast<size_t>(k)) || deg == 0) {
                    continue;
                }

                double a[3] = { 0, 0, 0 };

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e)
                    for (int c = 0; c < 3; ++c) {
                        a[c] += q[3 * static_cast<size_t>(sg.nb[e]) + c];
                    }

                for (int c = 0; c < 3; ++c) {
                    t[3 * static_cast<size_t>(k) + c] = a[c] / static_cast<double>(deg);
                }
            }

            q.swap(t);
        }

        if (p.verbose) {
            std::fprintf(stderr, "[infl]    unfold %d passes, %zu folded\n", pass, left);
        }

        return left;
    };
    std::vector<double> area0, encl0, area, encl;
    comp_area(y, area0);
    comp_volume(y, encl0);
    st.area = std::accumulate(area0.begin(), area0.end(), 0.0);

    for (size_t c = 0; c < ncs; ++c)
        if (closed[c]) {
            st.volume += std::fabs(encl0[c]);
            ++st.closed;
        }

    // the edges' first lengths (the tangential relaxation's rest lengths), per CSR entry
    std::vector<double> rest(sg.nb.size());
    #pragma omp parallel for schedule(static)

    for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s)
        for (int64_t e = sg.off[s]; e < sg.off[s + 1]; ++e) {
            const double* a = &y[3 * static_cast<size_t>(s)];
            const double* b = &y[3 * static_cast<size_t>(sg.nb[e])];
            rest[e] = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
        }

    phase("surface graph, areas");

    // 2. the moving mesh: its graph, first volumes, free nodes, node -> tets
    Csr tg;
    std::vector<double> vol0;
    std::vector<int32_t> fid(n, -1), free_nodes;
    std::vector<int64_t> n2t_off;
    std::vector<int32_t> n2t;

    if (move) {
        tg = build_csr(n, m.tets, 4);
        vol0.resize(nt);
        #pragma omp parallel for schedule(static)

        for (int64_t t = 0; t < static_cast<int64_t>(nt); ++t) {
            vol0[t] = tet_volume(x, &m.tets[4 * t]);
        }

        for (size_t v = 0; v < n; ++v)
            if (!fixed[v] && !issurf[v]) {
                fid[v] = static_cast<int32_t>(free_nodes.size());
                free_nodes.push_back(static_cast<int32_t>(v));
            }

        st.fixed_nodes = static_cast<size_t>(std::count(fixed.begin(), fixed.end(), 1));
        st.free_nodes = free_nodes.size();
        n2t_off.assign(n + 1, 0);

        for (size_t i = 0; i < 4 * nt; ++i) {
            ++n2t_off[static_cast<size_t>(m.tets[i]) + 1];
        }

        for (size_t v = 0; v < n; ++v) {
            n2t_off[v + 1] += n2t_off[v];
        }

        n2t.resize(4 * nt);
        std::vector<int64_t> cur(n2t_off.begin(), n2t_off.end() - 1);

        for (size_t i = 0; i < 4 * nt; ++i) {
            n2t[static_cast<size_t>(cur[static_cast<size_t>(m.tets[i])]++)] = static_cast<int32_t>(i / 4);
        }
    }

    phase("mesh setup");

    // the free nodes at the weighted-harmonic minimizer (w = 1 / current length), the others
    // held: Jacobi-preconditioned CG on the three coordinates at once, warm-started
    auto harmonic = [&](std::vector<double>& q) {
        const size_t nfree = free_nodes.size();

        if (nfree == 0) {
            return;
        }

        std::vector<double> w(tg.nb.size()), diag(nfree), b(3 * nfree, 0.0);
        #pragma omp parallel for schedule(static)

        for (int64_t i = 0; i < static_cast<int64_t>(nfree); ++i) {
            const int32_t v = free_nodes[static_cast<size_t>(i)];
            const double* a = P3(q, v);
            double d = 0.0;

            for (int64_t e = tg.off[v]; e < tg.off[v + 1]; ++e) {
                const int32_t u = tg.nb[e];
                const double* c = P3(q, u);
                const double l = std::sqrt((c[0] - a[0]) * (c[0] - a[0]) + (c[1] - a[1]) * (c[1] - a[1]) +
                                           (c[2] - a[2]) * (c[2] - a[2]));
                w[e] = 1.0 / std::max(l, 1e-6);
                d += w[e];

                if (fid[static_cast<size_t>(u)] < 0)
                    for (int k = 0; k < 3; ++k) {
                        b[3 * i + k] += w[e] * c[k];
                    }
            }

            diag[i] = d;
        }

        auto apply = [&](const std::vector<double>& in, std::vector<double>& out) {
            #pragma omp parallel for schedule(static)

            for (int64_t i = 0; i < static_cast<int64_t>(nfree); ++i) {
                const int32_t v = free_nodes[static_cast<size_t>(i)];
                double s[3] = { diag[i] * in[3 * i], diag[i] * in[3 * i + 1], diag[i] * in[3 * i + 2] };

                for (int64_t e = tg.off[v]; e < tg.off[v + 1]; ++e) {
                    const int32_t j = fid[static_cast<size_t>(tg.nb[e])];

                    if (j >= 0)
                        for (int k = 0; k < 3; ++k) {
                            s[k] -= w[e] * in[3 * static_cast<size_t>(j) + k];
                        }
                }

                for (int k = 0; k < 3; ++k) {
                    out[3 * i + k] = s[k];
                }
            }
        };
        std::vector<double> xf(3 * nfree), r(3 * nfree), z(3 * nfree), dq(3 * nfree), Ap(3 * nfree);

        for (size_t i = 0; i < nfree; ++i)
            for (int k = 0; k < 3; ++k) {
                xf[3 * i + k] = q[3 * static_cast<size_t>(free_nodes[i]) + k];
            }

        pcg3(nfree, diag, apply, b, xf, 1e-5, 2000);

        for (size_t i = 0; i < nfree; ++i)
            for (int k = 0; k < 3; ++k) {
                q[3 * static_cast<size_t>(free_nodes[i]) + k] = xf[3 * i + k];
            }
    };

    auto ratio = [&](const std::vector<double>& q, int32_t t) {
        return vol0[static_cast<size_t>(t)] != 0.0 ? tet_volume(q, &m.tets[4 * static_cast<size_t>(t)]) / vol0[static_cast<size_t>(t)]
               : 1.0;
    };
    auto min_ratio = [&](const std::vector<double>& q) {   // the smallest tet volume over its first (<= 0: inverted)
        double r = 1e30;
        #pragma omp parallel for reduction(min : r) schedule(static)

        for (int64_t t = 0; t < static_cast<int64_t>(nt); ++t) {
            r = std::min(r, ratio(q, static_cast<int32_t>(t)));
        }

        return r;
    };
    auto local_min = [&](const std::vector<double>& q, int32_t v) {
        double r = 1e30;

        for (int64_t e = n2t_off[static_cast<size_t>(v)]; e < n2t_off[static_cast<size_t>(v) + 1]; ++e) {
            r = std::min(r, ratio(q, n2t[static_cast<size_t>(e)]));
        }

        return r;
    };
    const double flat = 0.02;   // (below 2% of its first volume a tet counts as nearly flat)
    auto penalty = [&](const std::vector<double>& q, int32_t v) {   // sum of (flat - r)^2 over v's flat tets
        double e = 0.0;

        for (int64_t k = n2t_off[static_cast<size_t>(v)]; k < n2t_off[static_cast<size_t>(v) + 1]; ++k) {
            const double r = ratio(q, n2t[static_cast<size_t>(k)]);

            if (r < flat) {
                e += (flat - r) * (flat - r);
            }
        }

        return e;
    };
    // nodes of the flat / inverted tets pushed out along their volume gradient (a free node, or
    // an inflated surface's own -- it gives way where its target would invert a tet with no
    // free node; the outer boundary never moves), a move kept when it lessens the flatness
    // without newly inverting a tet; returns the tets still inverted
    auto untangle = [&](std::vector<double>& q) {
        std::vector<int32_t> badt;
        std::vector<char> tmark(nt, 0), vmark(n, 0);

        {   // (the scan in parallel, the few found collected after)
            std::vector<char> fl(nt);
            #pragma omp parallel for schedule(static)

            for (int64_t t = 0; t < static_cast<int64_t>(nt); ++t) {
                fl[static_cast<size_t>(t)] = ratio(q, static_cast<int32_t>(t)) <= flat;
            }

            for (size_t t = 0; t < nt; ++t)
                if (fl[t]) {
                    badt.push_back(static_cast<int32_t>(t));
                }
        }

        for (int round = 0; round < 200 && !badt.empty(); ++round) {
            std::vector<int32_t> cand, moved;

            for (int32_t t : badt)
                for (int k = 0; k < 4; ++k) {
                    const int32_t v = m.tets[4 * static_cast<size_t>(t) + k];

                    if (!fixed[static_cast<size_t>(v)] && !vmark[static_cast<size_t>(v)]) {
                        vmark[static_cast<size_t>(v)] = 1;
                        cand.push_back(v);
                    }
                }

            for (int32_t v : cand) {
                vmark[static_cast<size_t>(v)] = 0;
                double step[3] = { 0, 0, 0 };
                int nb = 0;

                for (int64_t e = n2t_off[static_cast<size_t>(v)]; e < n2t_off[static_cast<size_t>(v) + 1]; ++e) {
                    const int32_t t = n2t[static_cast<size_t>(e)];
                    const double r = ratio(q, t);

                    if (r > flat) {
                        continue;
                    }

                    // d(volume)/d(this corner) = -(the opposite face's cross product) / 6
                    const int32_t* tv = &m.tets[4 * static_cast<size_t>(t)];
                    int c = 0;

                    while (tv[c] != v) {
                        ++c;
                    }

                    const double* A = P3(q, tv[kFaceOf[c][0]]);
                    const double* B = P3(q, tv[kFaceOf[c][1]]);
                    const double* C = P3(q, tv[kFaceOf[c][2]]);
                    const double u[3] = { B[0] - A[0], B[1] - A[1], B[2] - A[2] };
                    const double w[3] = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };
                    const double g[3] = { -(u[1] * w[2] - u[2] * w[1]) / 6.0, -(u[2] * w[0] - u[0] * w[2]) / 6.0,
                                          -(u[0] * w[1] - u[1] * w[0]) / 6.0
                                        };
                    const double gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];

                    if (gg <= 0.0) {
                        continue;
                    }

                    // (a Newton step to 2 x flat of the first volume, in the first volume's sign)
                    const double sgn = vol0[static_cast<size_t>(t)] > 0.0 ? 1.0 : -1.0;
                    const double need = (2.0 * flat - r) * std::fabs(vol0[static_cast<size_t>(t)]);

                    for (int k = 0; k < 3; ++k) {
                        step[k] += sgn * need * g[k] / gg;
                    }

                    ++nb;
                }

                if (nb == 0) {
                    continue;
                }

                const double before = penalty(q, v), lmin = local_min(q, v);
                double* qv = &q[3 * static_cast<size_t>(v)];
                const double old[3] = { qv[0], qv[1], qv[2] };
                bool ok = false;

                for (double h = 1.0 / nb; h > 1e-3; h *= 0.5) {
                    for (int k = 0; k < 3; ++k) {
                        qv[k] = old[k] + h * step[k];
                    }

                    if (penalty(q, v) < before && (local_min(q, v) > 0.0 || local_min(q, v) >= lmin)) {
                        ok = true;
                        break;
                    }
                }

                if (ok) {
                    moved.push_back(v);
                } else {
                    for (int k = 0; k < 3; ++k) {
                        qv[k] = old[k];
                    }
                }
            }

            if (moved.empty()) {
                break;
            }

            // next round: the tets still flat, and the moved nodes' flat ones
            std::vector<int32_t> next;

            for (int32_t t : badt)
                if (!tmark[static_cast<size_t>(t)] && ratio(q, t) <= flat) {
                    tmark[static_cast<size_t>(t)] = 1;
                    next.push_back(t);
                }

            for (int32_t v : moved)
                for (int64_t e = n2t_off[static_cast<size_t>(v)]; e < n2t_off[static_cast<size_t>(v) + 1]; ++e) {
                    const int32_t t = n2t[static_cast<size_t>(e)];

                    if (!tmark[static_cast<size_t>(t)] && ratio(q, t) <= flat) {
                        tmark[static_cast<size_t>(t)] = 1;
                        next.push_back(t);
                    }
                }

            for (int32_t t : next) {
                tmark[static_cast<size_t>(t)] = 0;
            }

            badt.swap(next);
        }

        size_t bad = 0;

        for (int32_t t : badt) {
            bad += ratio(q, t) <= 0.0;
        }

        return bad;
    };

    // the surfaces into the mesh (a node's copies averaged) and back
    auto scatter = [&](const std::vector<double>& q) {
        std::vector<double> acc(3 * n, 0.0);
        std::vector<int32_t> cnt(n, 0);

        for (size_t k = 0; k < ns; ++k) {
            const size_t v = static_cast<size_t>(sv[k]);

            for (int c = 0; c < 3; ++c) {
                acc[3 * v + c] += q[3 * k + c];
            }

            ++cnt[v];
        }

        for (size_t v = 0; v < n; ++v)
            if (cnt[v] && !fixed[v])
                for (int c = 0; c < 3; ++c) {
                    x[3 * v + c] = acc[3 * v + c] / cnt[v];
                }
    };
    auto gather = [&](std::vector<double>& q) {
        for (size_t k = 0; k < ns; ++k)
            for (int c = 0; c < 3; ++c) {
                q[3 * k + c] = x[3 * static_cast<size_t>(sv[k]) + c];
            }
    };

    // FreeSurfer's sulc: each step's move along the (unit, outward) normal before it, summed
    std::vector<double> sacc(ns, 0.0), ybef, nbef;

    // 3. the steps
    std::vector<double> ny(3 * ns), nrm, ysnap, xprev;
    const double lambda = 0.5;

    if (move) {
        xprev = x;
    }

    // an implicit step of y into ny: (M + tau' L) Y = M Y_old (see the weights below)
    auto implicit = [&](double tau, bool cot) {
        // the weights: uniform (1 per edge, unit mass) or cotangent (Desbrun et al.: an edge
        // (cot a + cot b) / 2 from its two faces' opposite angles, clamped positive; a vertex's
        // mass a third of its faces' area; tau scaled by the median edge length squared), from
        // the current geometry -- cotangent moves the vertices along the normal (mean
        // curvature flow), not sideways into slivers and dimples as the uniform one does
        std::vector<double> w(sg.nb.size(), 1.0), mass(ns, 1.0);
        double sc = tau;

        if (cot) {
            std::fill(w.begin(), w.end(), 0.0);
            std::fill(mass.begin(), mass.end(), 0.0);
            auto slot = [&](size_t i, int32_t j) {   // (j's entry in i's sorted neighbour list)
                const int32_t* b0 = sg.nb.data() + sg.off[i];
                const int32_t* b1 = sg.nb.data() + sg.off[i + 1];
                return static_cast<size_t>(std::lower_bound(b0, b1, j) - sg.nb.data());
            };

            for (size_t f = 0; f < nf; ++f) {
                const int32_t v[3] = { sf[3 * f], sf[3 * f + 1], sf[3 * f + 2] };
                const double ar = face_area(y, f);

                for (int c = 0; c < 3; ++c) {
                    mass[static_cast<size_t>(v[c])] += ar / 3.0;
                    const double* o = &y[3 * static_cast<size_t>(v[c])];   // the corner opposite edge (a, b)
                    const int32_t a = v[(c + 1) % 3], bq = v[(c + 2) % 3];
                    const double* pa = &y[3 * static_cast<size_t>(a)];
                    const double* pb = &y[3 * static_cast<size_t>(bq)];
                    const double u[3] = { pa[0] - o[0], pa[1] - o[1], pa[2] - o[2] };
                    const double t[3] = { pb[0] - o[0], pb[1] - o[1], pb[2] - o[2] };
                    const double dot = u[0] * t[0] + u[1] * t[1] + u[2] * t[2];
                    const double cx = u[1] * t[2] - u[2] * t[1], cy = u[2] * t[0] - u[0] * t[2], cz = u[0] * t[1] - u[1] * t[0];
                    const double cr = std::sqrt(cx * cx + cy * cy + cz * cz);
                    const double ct = cr > 1e-20 ? 0.5 * dot / cr : 0.0;
                    w[slot(static_cast<size_t>(a), bq)] += ct;
                    w[slot(static_cast<size_t>(bq), a)] += ct;
                }
            }

            for (double& x_ : w) {   // (obtuse pairs give negative weights: clamped, the matrix kept definite)
                x_ = std::max(x_, 1e-3);
            }

            std::vector<double> rl(rest);
            std::nth_element(rl.begin(), rl.begin() + rl.size() / 2, rl.end());
            const double h = rl.empty() ? 1.0 : rl[rl.size() / 2];
            sc = tau * h * h;

            for (double& m_ : mass) {
                m_ = std::max(m_, 1e-12);
            }
        }

        std::vector<double> b(3 * ns), diag(ns);

        for (size_t k = 0; k < ns; ++k) {
            double ws = 0.0;

            for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e) {
                ws += w[static_cast<size_t>(e)];
            }

            diag[k] = held(k) ? 1.0 : mass[k] + sc * ws;

            for (int c = 0; c < 3; ++c) {
                ny[3 * k + c] = y[3 * k + c];
                b[3 * k + c] = (held(k) ? 1.0 : mass[k]) * y[3 * k + c];
            }

            if (!held(k))
                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e)
                    if (held(static_cast<size_t>(sg.nb[e])))
                        for (int c = 0; c < 3; ++c) {
                            b[3 * k + c] += sc * w[static_cast<size_t>(e)] * y[3 * static_cast<size_t>(sg.nb[e]) + c];
                        }
        }

        auto apply = [&](const std::vector<double>& in, std::vector<double>& out) {
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                for (int c = 0; c < 3; ++c) {
                    out[3 * k + c] = diag[k] * in[3 * k + c];
                }

                if (held(static_cast<size_t>(k))) {
                    continue;
                }

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e) {
                    const size_t j = static_cast<size_t>(sg.nb[e]);

                    if (!held(j))
                        for (int c = 0; c < 3; ++c) {
                            out[3 * k + c] -= sc * w[static_cast<size_t>(e)] * in[3 * j + c];
                        }
                }
            }
        };
        const int its = pcg3(ns, diag, apply, b, ny, 1e-6, 1000);

        if (p.verbose) {
            std::fprintf(stderr, "[infl]    cg %d iterations (%zu nodes, tau %g)\n", its, ns, sc);
        }
    };

    for (int it = 0; it < p.iters; ++it) {
        if (p.sulc_fs) {
            ybef = y;
            normals(y, nbef);
        }

        // (a) smoothing: implicit, (I + tau L) Y = Y_old, Jacobi-preconditioned CG (L = D - A,
        // held vertices fixed: one step smooths over ~sqrt(tau) edges); or an explicit
        // Laplacian step (+ its displacement averaged, FreeSurfer's n_averages)
        if (p.tau > 0.0) {
            implicit(p.tau, p.cot);
        } else {
            #pragma omp parallel for schedule(static)

            for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
                const int64_t deg = sg.off[s + 1] - sg.off[s];
                double lap[3] = { 0, 0, 0 };

                for (int64_t e = sg.off[s]; e < sg.off[s + 1]; ++e)
                    for (int c = 0; c < 3; ++c) {
                        lap[c] += y[3 * static_cast<size_t>(sg.nb[e]) + c] - y[3 * static_cast<size_t>(s) + c];
                    }

                for (int c = 0; c < 3; ++c) {
                    ny[3 * s + c] = y[3 * s + c] + (held(static_cast<size_t>(s)) || deg == 0 ? 0.0 : lambda * lap[c] / deg);
                }
            }

            if (p.averages > 0) {
                std::vector<double> dsp(3 * ns), dn(3 * ns);

                for (size_t i = 0; i < 3 * ns; ++i) {
                    dsp[i] = ny[i] - y[i];
                }

                for (int pass = 0; pass < p.averages; ++pass) {
                    #pragma omp parallel for schedule(static)

                    for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
                        const int64_t deg = sg.off[s + 1] - sg.off[s];
                        double acc[3] = { dsp[3 * s], dsp[3 * s + 1], dsp[3 * s + 2] };

                        for (int64_t e = sg.off[s]; e < sg.off[s + 1]; ++e)
                            for (int c = 0; c < 3; ++c) {
                                acc[c] += dsp[3 * static_cast<size_t>(sg.nb[e]) + c];
                            }

                        for (int c = 0; c < 3; ++c) {
                            dn[3 * s + c] = held(static_cast<size_t>(s)) ? 0.0 : acc[c] / (deg + 1);
                        }
                    }

                    dsp.swap(dn);
                }

                for (size_t i = 0; i < 3 * ns; ++i) {
                    ny[i] = y[i] + dsp[i];
                }
            }
        }

        y.swap(ny);
        lap(0);

        // (b) the edges relaxed toward their first lengths -- scaled by the surface's area
        // ratio when it keeps its volume (only the area's distribution corrected) -- moving
        // in the tangent plane only: the sulcal walls, squeezed by the smoothing, get their
        // share of the surface back, the shape stays smooth
        if (p.dist > 0.0 && p.relax > 0) {
            comp_area(y, area);
            std::vector<double> scl(ncs, 1.0);

            for (size_t c = 0; c < ncs; ++c) {
                scl[c] = !by_area && area0[c] > 0.0 ? std::sqrt(area[c] / area0[c]) : 1.0;
            }

            for (int pass = 0; pass < p.relax; ++pass) {
                normals(y, nrm);
                #pragma omp parallel for schedule(static)

                for (int64_t s = 0; s < static_cast<int64_t>(ns); ++s) {
                    const int64_t deg = sg.off[s + 1] - sg.off[s];

                    if (held(static_cast<size_t>(s)) || deg == 0) {
                        for (int c = 0; c < 3; ++c) {
                            ny[3 * s + c] = y[3 * s + c];
                        }

                        continue;
                    }

                    const double* a = &y[3 * static_cast<size_t>(s)];
                    const double L = scl[static_cast<size_t>(comp[static_cast<size_t>(s)])];
                    double f[3] = { 0, 0, 0 };

                    for (int64_t e = sg.off[s]; e < sg.off[s + 1]; ++e) {
                        const double* b = &y[3 * static_cast<size_t>(sg.nb[e])];
                        const double d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
                        const double l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

                        if (l > 1e-12)
                            for (int c = 0; c < 3; ++c) {
                                f[c] += (l - L * rest[e]) / l * d[c];
                            }
                    }

                    const double* nn = &nrm[3 * static_cast<size_t>(s)];
                    const double fn = f[0] * nn[0] + f[1] * nn[1] + f[2] * nn[2];
                    double to[3];

                    for (int c = 0; c < 3; ++c) {
                        to[c] = a[c] + p.dist * (f[c] - fn * nn[c]) / static_cast<double>(deg);
                    }

                    // (kept only if none of its faces turns over against its normal)
                    bool ok = true;

                    for (int64_t e = vfo[static_cast<size_t>(s)]; e < vfo[static_cast<size_t>(s) + 1] && ok; ++e) {
                        double fc[3];
                        face_n(y, static_cast<size_t>(vfl[static_cast<size_t>(e)]), to, static_cast<size_t>(s), fc);
                        ok = fc[0] * nn[0] + fc[1] * nn[1] + fc[2] * nn[2] > 0.0;
                    }

                    for (int c = 0; c < 3; ++c) {
                        ny[3 * s + c] = ok ? to[c] : a[c];
                    }
                }

                y.swap(ny);
            }
        }

        lap(1);
        // (b') the folds the smoothing left (fine folds squeezed over) smoothed out
        st.folds = unfold(y);
        lap(2);

        if (p.sulc_fs) {   // (the move by the smoothing, relaxation and unfolding: not the rescale's)
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                const size_t j = 3 * static_cast<size_t>(k);
                sacc[static_cast<size_t>(k)] += (y[j] - ybef[j]) * nbef[j] + (y[j + 1] - ybef[j + 1]) * nbef[j + 1] +
                                                (y[j + 2] - ybef[j + 2]) * nbef[j + 2];
            }
        }

        // (c) every closed surface back to its first area or volume, about its centroid (a
        // collapsed one -- a small island below 0.1% of it -- left alone, a step's factor
        // bounded: 1 / (almost 0) would blow it up); an open one keeps its area
        comp_area(y, area);
        comp_volume(y, encl);
        std::vector<double> cen(3 * ncs, 0.0), cnt(ncs, 0.0), sc(ncs, 1.0);

        for (size_t s = 0; s < ns; ++s) {
            for (int c = 0; c < 3; ++c) {
                cen[3 * static_cast<size_t>(comp[s]) + c] += y[3 * s + c];
            }

            cnt[static_cast<size_t>(comp[s])] += 1.0;
        }

        for (size_t c = 0; c < ncs; ++c) {
            const bool byvol = !by_area && closed[c] && encl[c] * encl0[c] > 0.0 && std::fabs(encl[c]) > 1e-3 * std::fabs(encl0[c]);
            const bool byar = !byvol && area[c] > 1e-3 * area0[c] && (by_area || !closed[c]);
            sc[c] = byvol ? std::cbrt(encl0[c] / encl[c]) : byar ? std::sqrt(area0[c] / area[c]) : 1.0;
            sc[c] = std::min(1.1, std::max(0.9, sc[c]));
        }

        for (size_t s = 0; s < ns; ++s) {
            const size_t c = static_cast<size_t>(comp[s]);

            if (held(s)) {
                continue;
            }

            for (int k = 0; k < 3; ++k) {
                const double ck = cen[3 * c + k] / cnt[c];
                y[3 * s + k] = ck + sc[c] * (y[3 * s + k] - ck);
            }
        }

        st.iters_done = it + 1;
        lap(3);

        // the sulc reference: the surface once smoothed by sulc_tau (implicit steps)
        if (p.tau > 0.0 && ysnap.empty() && p.tau * (it + 1) >= p.sulc_tau) {
            ysnap = y;
        }

        // (d) the moving mesh follows: the free nodes harmonic, untangled; where tets stay
        // inverted the step is taken back there only (their nodes and one ring), untangled
        // again; the whole step bisected as a last resort; stopped when even 1/1024 fails
        if (move && ((it + 1) % std::max(1, p.substep) == 0 || it + 1 == p.iters)) {
            scatter(y);
            harmonic(x);
            untangle(x);

            for (int back = 0; back < 10 && min_ratio(x) <= 0.0; ++back) {
                std::vector<char> hold(n, 0);

                for (size_t t = 0; t < nt; ++t)
                    if (ratio(x, static_cast<int32_t>(t)) <= 0.0)
                        for (int k = 0; k < 4; ++k) {
                            const int32_t v = m.tets[4 * t + k];
                            hold[static_cast<size_t>(v)] = 1;

                            for (int64_t e = tg.off[v]; e < tg.off[v + 1]; ++e) {
                                hold[static_cast<size_t>(tg.nb[e])] = 1;
                            }
                        }

                size_t nh = 0;

                for (size_t v = 0; v < n; ++v)
                    if (hold[v]) {
                        ++nh;

                        for (int k = 0; k < 3; ++k) {
                            x[3 * v + k] = xprev[3 * v + k];
                        }
                    }

                const size_t lb = untangle(x);

                if (p.verbose) {
                    std::fprintf(stderr, "[infl]  step %4d: %zu nodes held back -> %zu tets inverted\n", it + 1, nh, lb);
                }
            }

            double a = 1.0;
            std::vector<double> q = x;

            while (min_ratio(q) <= 0.0 && a > 1.0 / 1024) {
                a *= 0.5;

                for (size_t i = 0; i < q.size(); ++i) {
                    q[i] = xprev[i] + a * (x[i] - xprev[i]);
                }

                untangle(q);
            }

            if (min_ratio(q) <= 0.0) {
                x = xprev;
                gather(y);
                st.stopped = true;
                st.iters_done = it + 1 - p.substep;
                break;
            }

            if (a < 1.0) {
                ++st.shortened;
                x = q;
            }

            gather(y);   // (the surfaces where the mesh let them go)
            xprev = x;

            if (p.verbose) {
                std::fprintf(stderr, "[infl]  step %4d: domain step %.3g, smallest tet volume ratio %.3g\n", it + 1, a, min_ratio(x));
            }

            lap(4);
        }
    }

    if (p.verbose) {
        std::fprintf(stderr, "[infl]  steps: smooth %.0f, relax %.0f, unfold %.0f, rescale %.0f, mesh %.0f ms\n", tacc[0], tacc[1],
                     tacc[2], tacc[3], tacc[4]);
    }

    // the finish (surfaces alone): small cotangent implicit steps -- mean curvature flow moves the
    // vertices along the normal only, so the needles and dimples the uniform steps leave go,
    // the triangles' spread (and the sulc pattern on it) stays -- then a few Taubin passes
    if (!move)
        for (int k = 0; k < p.cot_finish; ++k) {
            implicit(p.cot_finish_tau, true);
            y.swap(ny);
        }

    // the finish: a few Taubin passes (lambda / mu, no shrinking) -- the small cones and dimples
    // an uneven triangulation leaves (a vertex a little above / below its fan) smoothed out,
    // the shape kept (a moving mesh's surfaces stay where the mesh let them)
    if (!move)
        for (int pass = 0; pass < 2 * p.finish; ++pass) {
            const double w = pass % 2 == 0 ? 0.5 : -0.53;
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                const int64_t deg = sg.off[k + 1] - sg.off[k];
                double a[3] = { 0, 0, 0 };

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e)
                    for (int c = 0; c < 3; ++c) {
                        a[c] += y[3 * static_cast<size_t>(sg.nb[e]) + c];
                    }

                for (int c = 0; c < 3; ++c) {
                    ny[3 * k + c] = held(static_cast<size_t>(k)) || deg == 0 ? y[3 * k + c]
                                    : y[3 * k + c] + w * (a[c] / static_cast<double>(deg) - y[3 * k + c]);
                }
            }

            y.swap(ny);
        }

    // 4. sulc: each vertex's displacement along the outward normal of a smoothed surface -- the
    // one smoothed by sulc_tau (the folds gone, the lobes kept; explicit steps: the final one)
    // --, high-passed, per surface about its median (half sulci, half gyri, as FreeSurfer's):
    // positive where the vertex lay below it (a sulcal fundus), negative where above (a crown)
    if (ysnap.empty()) {
        ysnap = y;
    }

    normals(ysnap, nrm);
    std::vector<double> sulc(ns);
    std::vector<std::vector<double>> per(ncs);

    for (size_t k = 0; k < ns; ++k) {
        sulc[k] = p.sulc_fs ? sacc[k]
                  : (ysnap[3 * k] - y0[3 * k]) * nrm[3 * k] + (ysnap[3 * k + 1] - y0[3 * k + 1]) * nrm[3 * k + 1] +
                  (ysnap[3 * k + 2] - y0[3 * k + 2]) * nrm[3 * k + 2];
    }

    // high-passed: minus its mean over ~sulc_hp rings (the lobes' shift out; the sulci stay)
    if (p.sulc_hp > 0) {
        std::vector<double> lo(sulc), t(ns);

        for (int pass = 0; pass < p.sulc_hp; ++pass) {
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                double a = lo[static_cast<size_t>(k)];

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e) {
                    a += lo[static_cast<size_t>(sg.nb[e])];
                }

                t[static_cast<size_t>(k)] = a / static_cast<double>(sg.off[k + 1] - sg.off[k] + 1);
            }

            lo.swap(t);
        }

        for (size_t k = 0; k < ns; ++k) {
            sulc[k] -= lo[k];
        }
    }

    for (size_t k = 0; k < ns; ++k) {
        per[static_cast<size_t>(comp[k])].push_back(sulc[k]);
    }

    // (the median by area, the final surface's: half its area sulci, half gyri -- by vertex
    // count, narrow gyral blades' many vertices would tip it)
    std::vector<double> va(ns, 0.0);

    for (size_t f = 0; f < nf; ++f) {
        const double ar = face_area(y, f) / 3.0;

        for (int j = 0; j < 3; ++j) {
            va[static_cast<size_t>(sf[3 * f + j])] += ar;
        }
    }

    std::vector<std::vector<std::pair<double, double>>> pw(ncs);

    for (size_t k = 0; k < ns; ++k) {
        pw[static_cast<size_t>(comp[k])].push_back({ sulc[k], va[k] });
    }

    std::vector<double> med(ncs, 0.0);

    for (size_t c = 0; c < ncs; ++c)
        if (!pw[c].empty()) {
            std::sort(pw[c].begin(), pw[c].end());
            double tot = 0.0, acc = 0.0;

            for (const auto& e : pw[c]) {
                tot += e.second;
            }

            for (const auto& e : pw[c]) {
                acc += e.second;

                if (acc >= 0.5 * tot) {
                    med[c] = e.first;
                    break;
                }
            }
        }

    (void)per;

    st.sulc_min = 1e30;
    st.sulc_max = -1e30;

    for (size_t k = 0; k < ns; ++k) {
        sulc[k] -= med[static_cast<size_t>(comp[k])];
        st.sulc_min = std::min(st.sulc_min, sulc[k]);
        st.sulc_max = std::max(st.sulc_max, sulc[k]);
    }

    // curv: the pre-smoothed surface's mean curvature (FreeSurfer's ?h.curv, of its white
    // surface): the umbrella (the neighbours' mean - the vertex) along the outward normal over
    // half the mean squared edge length -- positive in the sulci (a fundus lies below its
    // ring), negative on the crowns; 1/mm
    std::vector<double> curv(ns, 0.0), n0v;
    normals(y0, n0v);
    #pragma omp parallel for schedule(static)

    for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
        const int64_t deg = sg.off[k + 1] - sg.off[k];

        if (deg == 0) {
            continue;
        }

        double a[3] = { 0, 0, 0 }, l2 = 0.0;
        const double* x0 = &y0[3 * static_cast<size_t>(k)];

        for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e) {
            const double* b = &y0[3 * static_cast<size_t>(sg.nb[e])];

            for (int c = 0; c < 3; ++c) {
                a[c] += b[c];
                l2 += (b[c] - x0[c]) * (b[c] - x0[c]);
            }
        }

        const double* nn = &n0v[3 * static_cast<size_t>(k)];
        double d = 0.0;

        for (int c = 0; c < 3; ++c) {
            d += (a[c] / static_cast<double>(deg) - x0[c]) * nn[c];
        }

        l2 /= static_cast<double>(deg);
        curv[static_cast<size_t>(k)] = l2 > 0.0 ? 4.0 * d / l2 : 0.0;   // (a sphere of radius R: 1/R)
    }

    {   // (averaged over curv_avg rings)
        std::vector<double> t(ns);

        for (int pass = 0; pass < p.curv_avg; ++pass) {
            #pragma omp parallel for schedule(static)

            for (int64_t k = 0; k < static_cast<int64_t>(ns); ++k) {
                double a = curv[static_cast<size_t>(k)];

                for (int64_t e = sg.off[k]; e < sg.off[k + 1]; ++e) {
                    a += curv[static_cast<size_t>(sg.nb[e])];
                }

                t[static_cast<size_t>(k)] = a / static_cast<double>(sg.off[k + 1] - sg.off[k] + 1);
            }

            curv.swap(t);
        }
    }

    // 5. each set's surface: its locals, inflated, their sulc and curv, and its faces -- and
    // first, with orig, the surface before inflating (its pre-smoothed nodes, GIFTI's
    // Anatomical), the inflated one <name>_inflated sharing its faces by a _DataLink_
    m.surfaces.clear();

    for (size_t k = 0; k < p.sets.size(); ++k) {
        Mesh::NamedSurface out;
        out.name = p.sets[k].name + "_inflated";
        // (JGIFTI's property names; GIFTI's metadata: the hemisphere, the surface, inflated)
        out.values = { { "SulcalDepth", {} }, { "Curvature", {} } };
        const InflateSet& ks = p.sets[k];
        auto ends = [&](const char* t) {
            const std::string e(t);
            return ks.name.size() >= e.size() && ks.name.compare(ks.name.size() - e.size(), e.size(), e) == 0;
        };

        if (ks.side != 0) {
            out.meta.push_back({ "AnatomicalStructurePrimary", ks.side < 0 ? "CortexLeft" : "CortexRight" });
        }

        if (ends("white") || ends("pial")) {
            out.meta.push_back({ "AnatomicalStructureSecondary", ends("white") ? "GrayWhite" : "Pial" });
        }

        out.meta.push_back({ "GeometricType", "Inflated" });

        for (size_t v = l0[k]; v < l0[k + 1]; ++v) {
            out.nodes.insert(out.nodes.end(), { y[3 * v], y[3 * v + 1], y[3 * v + 2] });
            out.values[0].second.push_back(sulc[v]);
            out.values[1].second.push_back(curv[v]);
        }

        for (size_t f = 3 * f0[k]; f < 3 * f0[k + 1]; ++f) {
            out.tris.push_back(sf[f] - static_cast<int32_t>(l0[k]));
        }

        if (p.orig) {
            Mesh::NamedSurface an = out;
            an.name = p.sets[k].name;
            an.meta.back().second = "Anatomical";

            for (size_t v = l0[k]; v < l0[k + 1]; ++v)
                for (int c = 0; c < 3; ++c) {
                    an.nodes[3 * (v - l0[k]) + c] = y0[3 * v + c];
                }

            out.tris.clear();
            out.tri_link = "$.GIFTIData." + an.name + ".MeshTri3.Data";
            m.surfaces.push_back(std::move(an));
        }

        m.surfaces.push_back(std::move(out));
    }

    if (move) {
        st.min_vol_ratio = min_ratio(x);
    }

    st.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

void inflate_mesh(Mesh& m, const InflateParams& p, InflateStats& st) {
    OmpThreadCap cap;   // (min(cores, 8) threads: oversubscribed, the many short loops slowed 2x)
    inflate_impl(m, p, st, nullptr);
}

}  // namespace tn
