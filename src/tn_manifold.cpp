// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_manifold.cpp -- see tn_manifold.h.

#include "tn_manifold.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>

namespace tn {

namespace {

// the face-neighbour of each tet face (4 t + f: the face opposite corner f), -1 none
std::vector<int64_t> face_neighbours(const std::vector<int32_t>& tets) {
    const size_t nt = tets.size() / 4;
    struct FK {
        int32_t a, b, c;
        int64_t tf;
    };
    std::vector<FK> k(4 * nt);

    for (size_t t = 0; t < nt; ++t)
        for (int f = 0; f < 4; ++f) {
            int32_t v[3], n = 0;

            for (int j = 0; j < 4; ++j)
                if (j != f) {
                    v[n++] = tets[4 * t + j];
                }

            std::sort(v, v + 3);
            k[4 * t + f] = { v[0], v[1], v[2], static_cast<int64_t>(4 * t + f) };
        }

    std::sort(k.begin(), k.end(), [](const FK & x, const FK & y) {
        return x.a != y.a ? x.a < y.a : x.b != y.b ? x.b < y.b : x.c != y.c ? x.c < y.c : x.tf < y.tf;
    });
    std::vector<int64_t> nb(4 * nt, -1);

    for (size_t i = 0; i + 1 < k.size(); ++i)
        if (k[i].a == k[i + 1].a && k[i].b == k[i + 1].b && k[i].c == k[i + 1].c) {
            nb[static_cast<size_t>(k[i].tf)] = k[i + 1].tf;
            nb[static_cast<size_t>(k[i + 1].tf)] = k[i].tf;
            ++i;
        }

    return nb;
}

struct Pinch {
    int32_t u, v, a, b;   // the edge (u < v), the label pair (a < b; 0 the exterior)
};

// the edges on more than two faces between one label pair. dirty (optional, per
// node): only the tets with a dirty node are scanned, and only the edges with a
// dirty end reported (their faces all lie in those tets: the counts are whole)
std::vector<Pinch> pinched(const std::vector<int32_t>& tets, const std::vector<int32_t>& lab, const std::vector<int64_t>& nb,
                           const std::vector<char>* dirty = nullptr) {
    const size_t nt = lab.size();
    std::vector<std::array<int32_t, 4>> ent;

    for (size_t t = 0; t < nt; ++t) {
        if (lab[t] <= 0) {
            continue;
        }

        if (dirty && !((*dirty)[static_cast<size_t>(tets[4 * t])] || (*dirty)[static_cast<size_t>(tets[4 * t + 1])] ||
                       (*dirty)[static_cast<size_t>(tets[4 * t + 2])] || (*dirty)[static_cast<size_t>(tets[4 * t + 3])])) {
            continue;
        }

        for (int f = 0; f < 4; ++f) {
            const int64_t n = nb[4 * t + f] >= 0 ? nb[4 * t + f] >> 2 : -1;
            const int32_t ln = n >= 0 ? std::max(0, lab[static_cast<size_t>(n)]) : 0;

            if (ln == lab[t] || (ln > 0 && static_cast<size_t>(n) < t)) {
                continue;   // not a boundary face, or counted from the other side
            }

            int32_t v[3], k = 0;

            for (int j = 0; j < 4; ++j)
                if (j != f) {
                    v[k++] = tets[4 * t + j];
                }

            const int32_t a = std::min(ln, lab[t]), b = std::max(ln, lab[t]);

            for (int e = 0; e < 3; ++e) {
                const int32_t p = v[e], q = v[(e + 1) % 3];
                ent.push_back({ { std::min(p, q), std::max(p, q), a, b } });
            }
        }
    }

    std::sort(ent.begin(), ent.end());
    std::vector<Pinch> out;

    for (size_t i = 0; i < ent.size();) {
        size_t j = i + 1;

        while (j < ent.size() && ent[j] == ent[i]) {
            ++j;
        }

        if (j - i > 2 && (!dirty || (*dirty)[static_cast<size_t>(ent[i][0])] || (*dirty)[static_cast<size_t>(ent[i][1])])) {
            out.push_back({ ent[i][0], ent[i][1], ent[i][2], ent[i][3] });
        }

        i = j;
    }

    return out;
}

// connected components of same-label tets (across faces)
std::vector<int32_t> components(const std::vector<int32_t>& lab, const std::vector<int64_t>& nb) {
    const size_t nt = lab.size();
    std::vector<int32_t> comp(nt, -1), stk;
    int32_t nc = 0;

    for (size_t s = 0; s < nt; ++s) {
        if (lab[s] <= 0 || comp[s] >= 0) {
            continue;
        }

        comp[s] = nc;
        stk.assign(1, static_cast<int32_t>(s));

        while (!stk.empty()) {
            const int32_t t = stk.back();
            stk.pop_back();

            for (int f = 0; f < 4; ++f) {
                const int64_t n = nb[4 * static_cast<size_t>(t) + f];

                if (n < 0) {
                    continue;
                }

                const size_t u = static_cast<size_t>(n >> 2);

                if (comp[u] < 0 && lab[u] == lab[static_cast<size_t>(t)]) {
                    comp[u] = nc;
                    stk.push_back(static_cast<int32_t>(u));
                }
            }
        }

        ++nc;
    }

    return comp;
}

}  // namespace

size_t count_pinched_edges(const std::vector<int32_t>& tets, const std::vector<int32_t>& labels) {
    return pinched(tets, labels, face_neighbours(tets)).size();
}

template <typename T>
void make_manifold(std::vector<int32_t>& tets, std::vector<int32_t>& labels, const std::vector<T>& X,
                   const std::vector<int>& nest, ManifoldStats& st) {
    const auto t0 = std::chrono::steady_clock::now();
    st = ManifoldStats();
    const size_t nt = labels.size();
    std::vector<int32_t>& lab = labels;
    const std::vector<int64_t> nb = face_neighbours(tets);
    // node -> tets
    size_t nn = 0;

    for (int32_t v : tets) {
        nn = std::max(nn, static_cast<size_t>(v) + 1);
    }

    std::vector<int64_t> off(nn + 1, 0);

    for (int32_t v : tets) {
        ++off[static_cast<size_t>(v) + 1];
    }

    for (size_t v = 0; v < nn; ++v) {
        off[v + 1] += off[v];
    }

    std::vector<int32_t> inc(static_cast<size_t>(off[nn]));
    {
        std::vector<int64_t> pos(off.begin(), off.end() - 1);

        for (size_t t = 0; t < nt; ++t)
            for (int j = 0; j < 4; ++j) {
                inc[static_cast<size_t>(pos[static_cast<size_t>(tets[4 * t + j])]++)] = static_cast<int32_t>(t);
            }
    }

    auto vol = [&](int32_t t) {
        const T* p[4];

        for (int j = 0; j < 4; ++j) {
            p[j] = &X[3 * static_cast<size_t>(tets[4 * static_cast<size_t>(t) + j])];
        }

        const double a[3] = { double(p[1][0] - p[0][0]), double(p[1][1] - p[0][1]), double(p[1][2] - p[0][2]) };
        const double b[3] = { double(p[2][0] - p[0][0]), double(p[2][1] - p[0][1]), double(p[2][2] - p[0][2]) };
        const double c[3] = { double(p[3][0] - p[0][0]), double(p[3][1] - p[0][1]), double(p[3][2] - p[0][2]) };
        return std::fabs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
    };
    auto nest_index = [&](int l) {
        const auto it = std::find(nest.begin(), nest.end(), l);
        return it == nest.end() ? -1 : static_cast<int>(it - nest.begin());
    };

    // a tet relabels once: a wedge with one already relabelled is not taken (no
    // flipping back and forth between two edges)
    std::vector<char> frozen(nt, 0);
    std::vector<char> dirty;

    // one round: every pinched edge opened by relabelling one wedge
    auto open_pinches = [&](const std::vector<Pinch>& pin) {
        std::vector<std::pair<int32_t, int32_t>> change;   // (tet, new label)
        std::vector<char> touched(nt, 0);

        for (const Pinch& p : pin) {
            // the tets round the edge, and their wedges (same label, sharing a face on the edge)
            std::vector<int32_t> ring;
            const int32_t* A = &inc[static_cast<size_t>(off[static_cast<size_t>(p.u)])], *Ae = &inc[static_cast<size_t>(off[static_cast<size_t>(p.u) + 1])];
            const int32_t* B = &inc[static_cast<size_t>(off[static_cast<size_t>(p.v)])], *Be = &inc[static_cast<size_t>(off[static_cast<size_t>(p.v) + 1])];
            std::set_intersection(A, Ae, B, Be, std::back_inserter(ring));
            ring.erase(std::remove_if(ring.begin(), ring.end(), [&](int32_t t) {
                return lab[static_cast<size_t>(t)] <= 0;
            }), ring.end());
            std::vector<int> root(ring.size());
            std::iota(root.begin(), root.end(), 0);
            std::function<int(int)> find = [&](int x) {
                return root[static_cast<size_t>(x)] == x ? x : root[static_cast<size_t>(x)] = find(root[static_cast<size_t>(x)]);
            };

            for (size_t i = 0; i < ring.size(); ++i) {
                const size_t t = static_cast<size_t>(ring[i]);

                for (int f = 0; f < 4; ++f) {
                    const int32_t opp = tets[4 * t + f];

                    if (opp == p.u || opp == p.v || nb[4 * t + f] < 0) {
                        continue;   // (a face on the edge: opposite a corner off it)
                    }

                    const int32_t n = static_cast<int32_t>(nb[4 * t + f] >> 2);
                    const auto it = std::lower_bound(ring.begin(), ring.end(), n);

                    if (it != ring.end() && *it == n && lab[static_cast<size_t>(n)] == lab[t]) {
                        const int x = find(static_cast<int>(i)), y = find(static_cast<int>(it - ring.begin()));

                        if (x != y) {
                            root[static_cast<size_t>(std::max(x, y))] = std::min(x, y);
                        }
                    }
                }
            }

            std::map<int, std::pair<int32_t, double>> wedge;   // root -> (label, volume; < 0: has a frozen tet)
            std::map<int, char> fz;

            for (size_t i = 0; i < ring.size(); ++i) {
                const int r = find(static_cast<int>(i));
                auto& w = wedge[r];
                w.first = lab[static_cast<size_t>(ring[i])];
                w.second += vol(ring[i]);
                fz[r] |= frozen[static_cast<size_t>(ring[i])];
            }

            // which label's wedge gives way, to which label
            int32_t from = -1, to = -1;

            if (p.a == 0) {
                from = p.b;
                to = 0;
            } else {
                const int ia = nest_index(p.a), ib = nest_index(p.b);

                if (ia >= 0 && ib >= 0 && ia != ib) {
                    from = ia < ib ? p.a : p.b;   // the outer label
                    to = ia < ib ? p.b : p.a;
                }
            }

            int best = -1;
            double bv = 1e300;
            std::map<int32_t, int> nw;

            for (const auto& kv : wedge) {
                ++nw[kv.second.first];
            }

            for (const auto& kv : wedge) {
                const int32_t l = kv.second.first;

                if ((from >= 0 ? l == from : (l == p.a || l == p.b)) && nw[l] >= 2 && !fz[kv.first] && kv.second.second < bv) {
                    bv = kv.second.second;
                    best = kv.first;
                }
            }

            if (best < 0) {
                continue;
            }

            const int32_t l = wedge[best].first, nl = from >= 0 ? to : (l == p.a ? p.b : p.a);

            for (size_t i = 0; i < ring.size(); ++i)
                if (find(static_cast<int>(i)) == best && !touched[static_cast<size_t>(ring[i])]) {
                    touched[static_cast<size_t>(ring[i])] = 1;
                    change.emplace_back(ring[i], nl);
                }
        }

        for (const auto& c : change) {
            if (c.second <= 0) {
                ++st.removed;
            } else {
                ++st.relabelled;
            }

            lab[static_cast<size_t>(c.first)] = c.second;
            frozen[static_cast<size_t>(c.first)] = 1;
        }

        // the next round looks only near what changed
        dirty.assign(nn, 0);

        for (const auto& c : change)
            for (int j = 0; j < 4; ++j) {
                dirty[static_cast<size_t>(tets[4 * static_cast<size_t>(c.first) + j])] = 1;
            }

        return change.size();
    };
    auto resolve = [&]() {
        dirty.clear();

        for (int r = 0; r < 100; ++r) {
            const std::vector<Pinch> pin = pinched(tets, lab, nb, dirty.empty() ? nullptr : &dirty);

            if (st.rounds == 0 && r == 0) {
                st.pinched_before = pin.size();
            }

            if (pin.empty() || open_pinches(pin) == 0) {
                break;
            }

            ++st.rounds;
        }
    };
    const std::vector<int32_t> lab0(lab);
    const std::vector<int32_t> pre = nest.empty() ? std::vector<int32_t>() : components(lab, nb);
    resolve();

    // with an order: a piece of an outer label cut off by the relabelling and
    // enclosed by inner labels only merges into its enclosure
    if (!nest.empty()) {
        const std::vector<int32_t> post = components(lab, nb);
        // each pre component -> the post components its unchanged tets fell in
        std::map<int32_t, std::map<int32_t, size_t>> split;

        for (size_t t = 0; t < nt; ++t)
            if (lab[t] > 0 && lab[t] == lab0[t] && nest_index(lab[t]) >= 0) {
                ++split[pre[t]][post[t]];
            }

        std::vector<char> cutoff;
        int32_t npost = 0;

        for (int32_t c : post) {
            npost = std::max(npost, c + 1);
        }

        cutoff.assign(static_cast<size_t>(npost), 0);

        for (const auto& kv : split) {
            if (kv.second.size() < 2) {
                continue;
            }

            int32_t keep = -1;
            size_t kn = 0;

            for (const auto& pc : kv.second)
                if (pc.second > kn) {
                    kn = pc.second;
                    keep = pc.first;
                }

            for (const auto& pc : kv.second)
                if (pc.first != keep) {
                    cutoff[static_cast<size_t>(pc.first)] = 1;
                }
        }

        // their neighbours across faces
        std::map<int32_t, std::map<int32_t, size_t>> around;

        for (size_t t = 0; t < nt; ++t) {
            if (lab[t] <= 0 || !cutoff[static_cast<size_t>(post[t])]) {
                continue;
            }

            for (int f = 0; f < 4; ++f) {
                const int64_t n = nb[4 * t + f] >= 0 ? nb[4 * t + f] >> 2 : -1;
                const int32_t ln = n >= 0 ? std::max(0, lab[static_cast<size_t>(n)]) : 0;

                if (ln != lab[t]) {
                    ++around[post[t]][ln];
                }
            }
        }

        std::vector<int32_t> into(static_cast<size_t>(npost), -1), clab(static_cast<size_t>(npost), -1);

        for (size_t t = 0; t < nt; ++t)
            if (lab[t] > 0) {
                clab[static_cast<size_t>(post[t])] = lab[t];
            }

        for (const auto& kv : around) {
            const int32_t l = clab[static_cast<size_t>(kv.first)];

            const int il = nest_index(l);
            bool enclosed = il >= 0;
            int32_t best = -1;
            size_t bn = 0;

            for (const auto& nl : kv.second) {
                const int in = nest_index(nl.first);
                enclosed = enclosed && nl.first > 0 && in > il;

                if (nl.second > bn) {
                    bn = nl.second;
                    best = nl.first;
                }
            }

            if (enclosed && best > 0) {
                into[static_cast<size_t>(kv.first)] = best;
                ++st.pockets;
            }
        }

        for (size_t t = 0; t < nt; ++t)
            if (lab[t] > 0 && into[static_cast<size_t>(post[t])] > 0) {
                lab[t] = into[static_cast<size_t>(post[t])];
                ++st.pocket_tets;
            }

        if (st.pockets) {
            resolve();   // (the merge can pinch again)
        }
    }

    st.pinched_after = pinched(tets, lab, nb).size();

    // the tets given to the exterior leave
    size_t k = 0;

    for (size_t t = 0; t < nt; ++t)
        if (lab[t] > 0) {
            for (int j = 0; j < 4; ++j) {
                tets[4 * k + j] = tets[4 * t + j];
            }

            lab[k++] = lab[t];
        }

    tets.resize(4 * k);
    lab.resize(k);
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

template void make_manifold<float>(std::vector<int32_t>&, std::vector<int32_t>&, const std::vector<float>&,
                                   const std::vector<int>&, ManifoldStats&);
template void make_manifold<double>(std::vector<int32_t>&, std::vector<int32_t>&, const std::vector<double>&,
                                    const std::vector<int>&, ManifoldStats&);

}  // namespace tn
