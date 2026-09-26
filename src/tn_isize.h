// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_isize.h -- interface element sizes (--isize): a size (mm) that applies only in
// the band next to an interface, while --size / --lsize keep setting the interiors,
// so that the gradient limit grades the mesh from fine interfaces to coarse
// interiors. Three forms, the most specific wins:
//   h        every interface
//   L:h      every interface of label L (0: the outer surface)
//   A:B:h    only the A|B interface
// A voxel next to labels A and B gets the A|B pair size if one is given, else the
// smaller of the A and B label sizes, else the global size (0 = none).

#ifndef TRUSSNET_ISIZE_H
#define TRUSSNET_ISIZE_H

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace tn {

struct InterfaceSizes {
    float glob = 0.0f;         // every interface (0 = none)
    std::vector<float> lab;    // per label, index = label (0 = none)
    std::vector<float> pair;   // (a, b, h) triples, a < b

    bool empty() const {
        if (glob > 0 || !pair.empty()) {
            return false;
        }

        for (size_t l = 0; l < lab.size(); ++l)
            if (lab[l] > 0) {
                return false;
            }

        return true;
    }

    // a < 0: every interface; b < 0: every interface of label a; else the a|b pair
    void add(int a, int b, double h) {
        if (!(h > 0)) {
            throw std::runtime_error("isize: sizes must be > 0");
        }

        if (a > 65535 || b > 65535 || (a < 0 && b >= 0)) {
            throw std::runtime_error("isize: bad label");
        }

        const float v = static_cast<float>(h);

        if (a < 0) {
            glob = v;
        } else if (b < 0) {
            if (static_cast<int>(lab.size()) <= a) {
                lab.resize(a + 1, 0.0f);
            }

            lab[a] = v;
        } else {
            if (a == b) {
                throw std::runtime_error("isize: a pair needs two different labels");
            }

            const float lo = static_cast<float>(std::min(a, b)), hi = static_cast<float>(std::max(a, b));

            for (size_t p = 0; p + 2 < pair.size(); p += 3)
                if (pair[p] == lo && pair[p + 1] == hi) {
                    pair[p + 2] = v;
                    return;
                }

            pair.push_back(lo);
            pair.push_back(hi);
            pair.push_back(v);
        }
    }

    // "h", "L:h", "A:B:h", comma-separated and mixed, e.g. "3,0:2,1:2:1"
    void parse(const std::string& s) {
        size_t p0 = 0;

        while (p0 <= s.size()) {
            const size_t p1 = s.find(',', p0);
            const std::string item = s.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
            std::vector<std::string> f;
            size_t q0 = 0;

            while (true) {
                const size_t q1 = item.find(':', q0);
                f.push_back(item.substr(q0, q1 == std::string::npos ? std::string::npos : q1 - q0));

                if (q1 == std::string::npos) {
                    break;
                }

                q0 = q1 + 1;
            }

            for (size_t k = 0; k < f.size(); ++k) {
                char* end = nullptr;
                std::strtod(f[k].c_str(), &end);

                if (f[k].empty() || *end != '\0') {
                    throw std::runtime_error("isize: bad item '" + item + "' (want H, L:H or A:B:H)");
                }
            }

            const double h = std::atof(f.back().c_str());

            if (f.size() == 1) {
                add(-1, -1, h);
            } else if (f.size() == 2) {
                add(std::atoi(f[0].c_str()), -1, h);
            } else if (f.size() == 3) {
                add(std::atoi(f[0].c_str()), std::atoi(f[1].c_str()), h);
            } else {
                throw std::runtime_error("isize: bad item '" + item + "' (want H, L:H or A:B:H)");
            }

            if (p1 == std::string::npos) {
                break;
            }

            p0 = p1 + 1;
        }
    }

    // the flattened (a, b, h) form of set_option: a = b = -1 global, b = -1 label a
    void add_triples(const std::vector<double>& v) {
        if (v.size() % 3) {
            throw std::runtime_error("isize wants (a, b, size) triples (-1 = any)");
        }

        for (size_t k = 0; k + 2 < v.size(); k += 3) {
            add(static_cast<int>(v[k]), static_cast<int>(v[k + 1]), v[k + 2]);
        }
    }

    // the size of the a|b interface (0 = none)
    float of(int a, int b) const {
        const float lo = static_cast<float>(std::min(a, b)), hi = static_cast<float>(std::max(a, b));

        for (size_t p = 0; p + 2 < pair.size(); p += 3)
            if (pair[p] == lo && pair[p + 1] == hi) {
                return pair[p + 2];
            }

        const float ha = a < static_cast<int>(lab.size()) ? lab[a] : 0.0f;
        const float hb = b < static_cast<int>(lab.size()) ? lab[b] : 0.0f;

        if (ha > 0 && hb > 0) {
            return std::min(ha, hb);
        }

        return ha > 0 ? ha : (hb > 0 ? hb : glob);
    }

    // widen [hmin, hmax] to the given sizes
    void span(float& hmin, float& hmax) const {
        auto f = [&](float v) {
            if (v > 0) {
                hmin = std::min(hmin, v);
                hmax = std::max(hmax, v);
            }
        };
        f(glob);

        for (size_t l = 0; l < lab.size(); ++l) {
            f(lab[l]);
        }

        for (size_t p = 2; p < pair.size(); p += 3) {
            f(pair[p]);
        }
    }
};

}  // namespace tn

#endif
