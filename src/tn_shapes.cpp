// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_shapes.cpp -- see tn_shapes.h.

#include "tn_shapes.h"

#include <cmath>
#include <functional>
#include <stdexcept>

namespace tn {

namespace {

const double kPi = 3.14159265358979323846;

LabelVolume rasterize(int n, const std::function<int(double, double, double)>& f) {
    LabelVolume lv;
    lv.nx = lv.ny = lv.nz = n;
    lv.data.resize(static_cast<size_t>(n) * n * n);
    lv.affine = { { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } };
    int mx = 0;

    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                // unit cube coordinates in (-1, 1), voxel centres
                const double x = 2.0 * (i + 0.5) / n - 1.0, y = 2.0 * (j + 0.5) / n - 1.0,
                             z = 2.0 * (k + 0.5) / n - 1.0;
                const int l = f(x, y, z);
                lv.data[i + static_cast<size_t>(n) * (j + static_cast<size_t>(n) * k)] = static_cast<uint16_t>(l);
                mx = l > mx ? l : mx;
            }

    lv.maxlabel = mx;
    return lv;
}

}  // namespace

std::vector<std::string> shape_names() {
    return { "sphere", "twoballs", "corrsphere", "gyroid", "torus", "ushape", "tjunction", "helix3", "boxhemi",
             "sandwich", "shells", "hollow", "slab" };
}

LabelVolume make_shape(const std::string& name, int n) {
    if (name == "sphere") {
        return rasterize(n, [](double x, double y, double z) {
            return x * x + y * y + z * z < 0.7 * 0.7 ? 1 : 0;
        });
    }

    if (name == "twoballs") {   // two labels meeting on a flat disk interface
        return rasterize(n, [](double x, double y, double z) {
            const double a = (x + 0.3) * (x + 0.3) + y * y + z * z, b = (x - 0.3) * (x - 0.3) + y * y + z * z;
            if (a < 0.45 * 0.45 && x < 0) {
                return 1;
            }
            if (b < 0.45 * 0.45 && x >= 0) {
                return 2;
            }
            return a < 0.45 * 0.45 ? 1 : (b < 0.45 * 0.45 ? 2 : 0);
        });
    }

    if (name == "corrsphere") {   // corrugated sphere r = R(1 + 0.12 sin 6th sin 6ph)
        return rasterize(n, [](double x, double y, double z) {
            const double r = std::sqrt(x * x + y * y + z * z);
            if (r < 1e-9) {
                return 1;
            }
            const double th = std::acos(z / r), ph = std::atan2(y, x);
            return r < 0.6 * (1.0 + 0.12 * std::sin(6 * th) * std::sin(6 * ph)) ? 1 : 0;
        });
    }

    if (name == "gyroid") {   // a gyroid sheet (thickness 0.5) inside a ball: very concave
        return rasterize(n, [](double x, double y, double z) {
            const double s = 2.0 * kPi * 1.2;
            const double gy = std::sin(s * x) * std::cos(s * y) + std::sin(s * y) * std::cos(s * z) +
                              std::sin(s * z) * std::cos(s * x);
            return (std::fabs(gy) < 0.5 && x * x + y * y + z * z < 0.85 * 0.85) ? 1 : 0;
        });
    }

    if (name == "torus") {
        return rasterize(n, [](double x, double y, double z) {
            const double q = std::sqrt(x * x + y * y) - 0.55;
            return q * q + z * z < 0.22 * 0.22 ? 1 : 0;
        });
    }

    if (name == "ushape") {   // a thick U-bend: a box with a slot cut from one side
        return rasterize(n, [](double x, double y, double z) {
            const bool box = std::fabs(x) < 0.7 && std::fabs(y) < 0.7 && std::fabs(z) < 0.35;
            const bool slot = std::fabs(x) < 0.3 && y > -0.3;
            return box && !slot ? 1 : 0;
        });
    }

    if (name == "tjunction") {   // three labels in a ball: z<0 -> 1; z>=0: x<0 -> 2, x>=0 -> 3
        return rasterize(n, [](double x, double y, double z) {
            if (x * x + y * y + z * z >= 0.75 * 0.75) {
                return 0;
            }
            return z < 0 ? 1 : (x < 0 ? 2 : 3);
        });
    }

    if (name == "helix3") {   // three interleaved helical tubes, one label each
        return rasterize(n, [](double x, double y, double z) {
            if (std::fabs(z) > 0.8) {
                return 0;
            }
            for (int t = 0; t < 3; ++t) {
                const double a = 2.0 * kPi * (z * 0.8 + t / 3.0);
                const double cx = 0.35 * std::cos(a), cy = 0.35 * std::sin(a);
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < 0.17 * 0.17) {
                    return t + 1;
                }
            }
            return 0;
        });
    }

    if (name == "boxhemi") {   // a box (1) with hemispheres (2, 3) on two opposite faces
        return rasterize(n, [](double x, double y, double z) {
            if (std::fabs(x) < 0.4 && std::fabs(y) < 0.4 && std::fabs(z) < 0.4) {
                return 1;
            }
            if (z >= 0.4 && x * x + y * y + (z - 0.4) * (z - 0.4) < 0.35 * 0.35) {
                return 2;
            }
            if (z <= -0.4 && x * x + y * y + (z + 0.4) * (z + 0.4) < 0.35 * 0.35) {
                return 3;
            }
            return 0;
        });
    }

    if (name == "sandwich") {   // a hemisphere cut into three stacked layers
        return rasterize(n, [](double x, double y, double z) {
            if (z < -0.6 || x * x + y * y + (z + 0.6) * (z + 0.6) >= 1.3 * 1.3 * 0.64) {
                return 0;
            }
            return z < -0.25 ? 1 : (z < 0.1 ? 2 : 3);
        });
    }

    if (name == "shells") {   // three nested spherical shells
        return rasterize(n, [](double x, double y, double z) {
            const double r = std::sqrt(x * x + y * y + z * z);
            return r < 0.35 ? 3 : (r < 0.55 ? 2 : (r < 0.75 ? 1 : 0));
        });
    }

    if (name == "hollow") {   // a hollow ball: the cavity is exterior (label 0)
        return rasterize(n, [](double x, double y, double z) {
            const double r = std::sqrt(x * x + y * y + z * z);
            return (r < 0.75 && r > 0.45) ? 1 : 0;
        });
    }

    if (name == "slab") {   // a thin slab, 6% of the box thick
        return rasterize(n, [](double x, double y, double z) {
            return (std::fabs(z) < 0.06 && std::fabs(x) < 0.7 && std::fabs(y) < 0.7) ? 1 : 0;
        });
    }

    throw std::runtime_error("unknown shape '" + name + "'");
}

}  // namespace tn
