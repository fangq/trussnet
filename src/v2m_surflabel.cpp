// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_surflabel.cpp -- see v2m_surflabel.h.

#include "v2m_surflabel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>

#include "v2m_cdt.h"
#include "v2m_modes.h"

namespace tn {

std::vector<int> solve_cell_labels(int ncells, const std::vector<int>& side, const std::vector<std::vector<int>>& labels,
                                   const std::vector<double>& vol, const SurfLabelOptions& o, SurfLabelStats& st) {
    const size_t nf = labels.size();
    std::vector<int> lab(static_cast<size_t>(std::max(ncells, 1)), -1);
    lab[0] = 0;
    bool any = false;

    for (const auto& L : labels) {
        any = any || !L.empty();
    }

    auto cvol = [&](int c) {
        return static_cast<size_t>(c) < vol.size() ? vol[static_cast<size_t>(c)] : 0.0;
    };

    if (!any) {
        // unlabelled: the depth of each cell (surfaces crossed from the exterior)
        std::vector<int> depth(lab.size(), -1);
        std::vector<std::vector<int>> nb(lab.size());

        for (size_t f = 0; f < nf; ++f) {
            const int a = side[2 * f], b = side[2 * f + 1];

            if (a >= 0 && b >= 0 && a != b) {
                nb[static_cast<size_t>(a)].push_back(b);
                nb[static_cast<size_t>(b)].push_back(a);
            }
        }

        std::vector<int> q(1, 0);
        depth[0] = 0;

        for (size_t h = 0; h < q.size(); ++h)
            for (int c : nb[static_cast<size_t>(q[h])])
                if (depth[static_cast<size_t>(c)] < 0) {
                    depth[static_cast<size_t>(c)] = depth[static_cast<size_t>(q[h])] + 1;
                    q.push_back(c);
                }

        for (auto& d : depth)
            if (d < 0) {
                d = 1;
            }

        // the winding number of each cell, if the faces' orientation gives one:
        // crossing a face against its normal (positive -> negative side) enters
        // one more surface. Outward shells -- nested, or touching themselves --
        // agree everywhere: the cells they wind 0 times are exterior (pockets,
        // cavities), and the winding is the depth. Where the counts disagree
        // around a cell (sheets between regions, oriented from one to the
        // other), the orientation says nothing: every cell is a region.
        std::vector<int> w(lab.size(), INT32_MIN);
        bool consistent = true;
        {
            std::vector<std::vector<std::pair<int, int>>> wn(lab.size());   // (neighbour, its winding - mine)

            for (size_t f = 0; f < nf; ++f) {
                const int a = side[2 * f], b = side[2 * f + 1];   // a: negative side (inside), b: positive

                if (a >= 0 && b >= 0 && a != b) {
                    wn[static_cast<size_t>(b)].push_back(std::make_pair(a, 1));
                    wn[static_cast<size_t>(a)].push_back(std::make_pair(b, -1));
                }
            }

            std::vector<int> qq(1, 0);
            w[0] = 0;

            for (size_t h = 0; h < qq.size() && consistent; ++h) {
                const int c = qq[h];

                for (const auto& e : wn[static_cast<size_t>(c)]) {
                    const int want = w[static_cast<size_t>(c)] + e.second;

                    if (w[static_cast<size_t>(e.first)] == INT32_MIN) {
                        w[static_cast<size_t>(e.first)] = want;
                        qq.push_back(e.first);
                    } else if (w[static_cast<size_t>(e.first)] != want) {
                        consistent = false;
                        break;
                    }
                }
            }

            int lo = 0, hi = 0;

            for (size_t c = 1; c < w.size(); ++c) {
                if (w[c] == INT32_MIN) {
                    consistent = false;
                } else {
                    lo = std::min(lo, w[c]);
                    hi = std::max(hi, w[c]);
                }
            }

            if (lo < 0 && hi > 0) {
                consistent = false;   // wound both ways
            }

            if (consistent && hi == 0 && lo == 0) {
                consistent = false;
            }

            if (consistent && lo < 0)   // oriented inward throughout
                for (auto& x : w) {
                    x = -x;
                }
        }

        if (consistent) {
            for (size_t c = 1; c < lab.size(); ++c) {
                depth[c] = w[c];
            }
        }

        if (o.auto_labels == "depth") {
            for (size_t c = 1; c < lab.size(); ++c) {
                lab[c] = depth[c];
            }
        } else {
            std::vector<int> order;

            for (int c = 1; c < ncells; ++c)
                if (depth[static_cast<size_t>(c)] > 0) {
                    order.push_back(c);
                } else {
                    lab[static_cast<size_t>(c)] = 0;   // wound 0 times: a pocket of the exterior
                }

            std::sort(order.begin(), order.end(), [&](int a, int b) {
                if (depth[static_cast<size_t>(a)] != depth[static_cast<size_t>(b)]) {
                    return depth[static_cast<size_t>(a)] < depth[static_cast<size_t>(b)];
                }

                return cvol(a) != cvol(b) ? cvol(a) > cvol(b) : a < b;
            });

            for (size_t k = 0; k < order.size(); ++k) {
                lab[static_cast<size_t>(order[k])] = static_cast<int>(k) + 1;
            }
        }

        for (size_t c = 1; c < lab.size(); ++c) {
            st.unresolved += lab[c] == 0;
        }

        st.oriented = consistent;
        return lab;
    }

    // labelled: outward from the exterior, a layer of cells at a time (so no cell
    // is decided by the order the faces come in)
    std::vector<char> unl(lab.size(), 0), notl(lab.size(), 0);

    for (;;) {
        for (;;) {
            std::vector<std::map<int, int>> votes(lab.size());

            for (size_t f = 0; f < nf; ++f) {
                const int a = side[2 * f], b = side[2 * f + 1];

                if (a < 0 || b < 0 || a == b) {
                    continue;
                }

                for (int s = 0; s < 2; ++s) {
                    const int x = s ? b : a, y = s ? a : b;

                    if (lab[static_cast<size_t>(x)] < 0 || lab[static_cast<size_t>(y)] >= 0) {
                        continue;
                    }

                    const std::vector<int>& L = labels[f];
                    bool gave = false;

                    for (int l : L)
                        if (l != lab[static_cast<size_t>(x)]) {
                            ++votes[static_cast<size_t>(y)][l];
                            gave = true;
                        }

                    if (!gave) {
                        (L.empty() ? unl : notl)[static_cast<size_t>(y)] = 1;
                    }
                }
            }

            bool any_new = false;

            for (size_t c = 0; c < lab.size(); ++c) {
                if (votes[c].empty()) {
                    continue;
                }

                int best = -1, bn = -1;

                for (const auto& kv : votes[c])
                    if (kv.second > bn) {   // (ties: the lower label; std::map is ordered)
                        bn = kv.second;
                        best = kv.first;
                    }

                st.conflicts += votes[c].size() > 1;
                lab[c] = best;
                any_new = true;
            }

            if (!any_new) {
                break;
            }
        }

        // cells reached only across their neighbour's own label: cavities (0);
        // they may lead on to more cells
        bool cav = false;

        for (size_t c = 1; c < lab.size(); ++c)
            if (lab[c] < 0 && notl[c] && !unl[c]) {
                lab[c] = 0;
                ++st.unresolved;
                cav = true;
            }

        if (!cav) {
            break;
        }
    }

    // what no label reached (unlabelled faces among labelled ones): regions of
    // their own, after the labels given, largest first
    int top = 0;

    for (int l : lab) {
        top = std::max(top, l);
    }

    std::vector<int> rest;

    for (int c = 1; c < ncells; ++c)
        if (lab[static_cast<size_t>(c)] < 0) {
            rest.push_back(c);
        }

    std::sort(rest.begin(), rest.end(), [&](int a, int b) {
        return cvol(a) != cvol(b) ? cvol(a) > cvol(b) : a < b;
    });

    for (int c : rest) {
        lab[static_cast<size_t>(c)] = ++top;
    }

    return lab;
}

namespace {

double extent_of(const std::vector<double>& X) {
    double ext = 0;

    for (int a = 0; a < 3; ++a) {
        double lo = 1e300, hi = -1e300;

        for (size_t v = 0; v < X.size() / 3; ++v) {
            lo = std::min(lo, X[3 * v + a]);
            hi = std::max(hi, X[3 * v + a]);
        }

        ext = std::max(ext, hi - lo);
    }

    return ext;
}

// nodes closer than tol -> one (the first); returns the map
std::vector<int32_t> weld_nodes(const std::vector<double>& X, double tol, size_t& welded) {
    const size_t nv = X.size() / 3;
    std::vector<int32_t> rep(nv);
    std::unordered_map<int64_t, std::vector<int32_t>> grid;
    auto key = [&](double x, double y, double z, int dx, int dy, int dz) {
        const int64_t i = static_cast<int64_t>(std::floor(x / tol)) + dx, j = static_cast<int64_t>(std::floor(y / tol)) + dy,
                      k = static_cast<int64_t>(std::floor(z / tol)) + dz;
        return (i * 73856093LL) ^ (j * 19349663LL) ^ (k * 83492791LL);
    };
    welded = 0;

    for (size_t v = 0; v < nv; ++v) {
        const double x = X[3 * v], y = X[3 * v + 1], z = X[3 * v + 2];
        int32_t r = -1;

        for (int dz = -1; dz <= 1 && r < 0; ++dz)
            for (int dy = -1; dy <= 1 && r < 0; ++dy)
                for (int dx = -1; dx <= 1 && r < 0; ++dx) {
                    auto it = grid.find(key(x, y, z, dx, dy, dz));

                    if (it == grid.end()) {
                        continue;
                    }

                    for (int32_t u : it->second) {
                        const double ex = X[3 * static_cast<size_t>(u)] - x, ey = X[3 * static_cast<size_t>(u) + 1] - y,
                                     ez = X[3 * static_cast<size_t>(u) + 2] - z;

                        if (ex * ex + ey * ey + ez * ez <= tol * tol) {
                            r = u;
                            break;
                        }
                    }
                }

        if (r < 0) {
            rep[v] = static_cast<int32_t>(v);
            grid[key(x, y, z, 0, 0, 0)].push_back(static_cast<int32_t>(v));
        } else {
            rep[v] = r;
            ++welded;
        }
    }

    return rep;
}

}  // namespace

bool normalize_surface_labels(Mesh& m, const SurfLabelOptions& o, SurfLabelStats& st) {
    const size_t nf = m.tris.size() / 3;
    st = SurfLabelStats();
    st.faces = nf;
    const bool labelled = m.tri_labels.size() == 2 * nf && nf > 0;
    bool pairs = false;

    if (labelled)
        for (size_t i = 0; i < nf && !pairs; ++i) {
            pairs = m.tri_labels[2 * i + 1] != 0;
        }

    st.kind = pairs ? "inner/outer labels" : labelled ? "one label per face" : "unlabelled";

    if (pairs || nf == 0) {
        st.note = pairs ? "as given" : "no faces";
        return false;
    }

    // 1. weld; repeated faces merged (their labels kept); degenerate ones dropped
    const double ext = extent_of(m.nodes);
    const std::vector<int32_t> rep = weld_nodes(m.nodes, 1e-7 * std::max(ext, 1e-300), st.welded);
    std::map<std::array<int32_t, 3>, size_t> seen;
    std::vector<std::array<int32_t, 3>> F;
    std::vector<std::vector<int>> L;

    for (size_t i = 0; i < nf; ++i) {
        std::array<int32_t, 3> t = { {
                rep[static_cast<size_t>(m.tris[3 * i])], rep[static_cast<size_t>(m.tris[3 * i + 1])],
                rep[static_cast<size_t>(m.tris[3 * i + 2])]
            }
        };

        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
            ++st.degenerate;
            continue;
        }

        std::array<int32_t, 3> k = t;
        std::sort(k.begin(), k.end());
        const int l = labelled ? m.tri_labels[2 * i] : 0;
        auto it = seen.find(k);

        if (it == seen.end()) {
            seen[k] = F.size();
            F.push_back(t);
            L.push_back(l > 0 ? std::vector<int>(1, l) : std::vector<int>());
        } else {
            ++st.duplicates;
            std::vector<int>& ls = L[it->second];

            if (l > 0 && std::find(ls.begin(), ls.end(), l) == ls.end()) {
                ls.push_back(l);
            }
        }
    }

    st.unique_faces = F.size();
    std::vector<int32_t> tris;
    tris.reserve(3 * F.size());

    for (const auto& t : F) {
        tris.insert(tris.end(), t.begin(), t.end());
    }

    // 2. closed, and not crossing itself?
    {
        std::unordered_map<uint64_t, int> ec;

        for (const auto& t : F)
            for (int k = 0; k < 3; ++k) {
                uint64_t a = static_cast<uint32_t>(t[k]), b = static_cast<uint32_t>(t[(k + 1) % 3]);

                if (a > b) {
                    std::swap(a, b);
                }

                ++ec[(a << 32) | b];
            }

        for (const auto& kv : ec) {
            st.open_edges += kv.second == 1;
        }
    }

    if (st.open_edges > 0) {
        st.note = "open surface: the rasterizer's rules";
        return false;
    }

    st.crossings = self_intersections(m.nodes, tris);

    if (st.crossings > 0) {
        st.note = "surfaces cross: the rasterizer's rules";
        return false;
    }

    // 3. the cells, and their labels
    Mesh clean;
    clean.nodes = m.nodes;
    clean.tris = tris;
    SurfCells sc;

    try {
        cdt_cells(clean, sc);
    } catch (const std::exception& e) {
        st.note = std::string("no cells (") + e.what() + "): the rasterizer's rules";
        return false;
    }

    for (size_t f = 0; f < F.size(); ++f)
        if (sc.side[2 * f] < 0 || sc.side[2 * f + 1] < 0) {
            st.note = "a face without its cells: the rasterizer's rules";
            return false;
        }

    st.cells = sc.ncells;
    const std::vector<int> lab = solve_cell_labels(sc.ncells, sc.side, L, sc.vol, o, st);
    // 4. inner / outer: the cells on the face's negative / positive side (the
    // normal inner -> outer); faces inside one region dropped
    m.tris.clear();
    m.tri_labels.clear();

    for (size_t f = 0; f < F.size(); ++f) {
        const int a = lab[static_cast<size_t>(sc.side[2 * f])], b = lab[static_cast<size_t>(sc.side[2 * f + 1])];

        if (a == b) {
            continue;
        }

        m.tris.insert(m.tris.end(), F[f].begin(), F[f].end());
        m.tri_labels.push_back(a);
        m.tri_labels.push_back(b);
    }

    st.exact = true;
    st.note = "inner/outer labels from the cells";
    return true;
}

std::string describe(const SurfLabelStats& st) {
    std::string s = "surfaces: " + st.kind + ", " + std::to_string(st.faces) + " faces";

    if (st.duplicates) {
        s += " (" + std::to_string(st.duplicates) + " repeated)";
    }

    if (st.welded) {
        s += ", " + std::to_string(st.welded) + " nodes welded";
    }

    if (st.open_edges) {
        s += ", " + std::to_string(st.open_edges) + " open edges";
    }

    if (st.crossings) {
        s += ", " + std::to_string(st.crossings) + " crossing pairs";
    }

    if (st.exact) {
        s += " -> " + std::to_string(st.cells - 1) + " enclosed cells";

        if (st.unresolved) {
            s += " (" + std::to_string(st.unresolved) + " cavities)";
        }

        if (st.kind == "unlabelled") {
            s += st.oriented ? ", by the faces' orientation" : ", each a region";
        }

        if (st.conflicts) {
            s += ", " + std::to_string(st.conflicts) + " with conflicting labels";
        }
    }

    return s + ": " + st.note;
}

}  // namespace tn
