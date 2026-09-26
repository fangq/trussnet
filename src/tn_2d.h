// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_2d.h -- the moving-particle mesher for 2-D images: a binary / multi-label
// image, or a gray-scale image with thresholds, to a conforming triangle mesh.
// The same algorithm as the 3-D pipeline in its own unit, on the CPU (OpenMP):
// smoothed interface fields (or gray-scale memberships), curvature sizing with
// gradient limiting (per-label sizes, a user sizing field), graded hexagonal
// seeding with fixed junction nodes where >= 3 labels meet, DistMesh spring
// relaxation with the nodes trapped on (and gliding along) the interface curves,
// an exact incremental Delaunay (Diazzi/Attene predicates), label-set tet labels
// and conformity repair.

#ifndef TRUSSNET_2D_H
#define TRUSSNET_2D_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tn_isize.h"

namespace tn {

struct Image2D {
    int nx = 0, ny = 0;
    std::array<double, 2> vs{ { 1, 1 } };                 // pixel size (mm)
    std::array<double, 6> affine{ { 1, 0, 0, 0, 1, 0 } };  // 2 x 3 row-major: pixel (i, j) -> world
    std::vector<uint16_t> lab;       // labels, x fastest (0 = exterior); filled from gray when thresholds
    std::vector<float> gray;         // gray-scale intensity (optional)
    std::vector<float> thresholds;   // gray-scale iso-values (ascending after meshing)
};

struct Mesh2DOptions {
    double size = 0, hmin = 0, hmax = 0;   // mm; 0 = 3 x pixel, size / 3, size
    double K = 3.0, grad = 0.3;            // curvature elements per radian; |grad h| limit
    double sigma = 0.5;                    // indicator smoothing (pixels; 1 erodes thin layers in 2-D)
    double gray_sigma = 0.0;               // gray-scale pre-smoothing (pixels)
    std::vector<float> hlab;               // per-label size (index = label), 0 = default
    InterfaceSizes isize;                  // interface sizes (mm; tn_isize.h), at interfaces only
    std::vector<float> hvox;               // user sizing field (per pixel, x fastest), 0 = automatic
    int nseed = 8;                         // lattice levels between hmin and hmax
    int iters = 300;                       // relaxation iterations (max)
    double fscale = 1.2, fsurf = 1.0;      // rest length / h (interior pairs, interface pairs)
    double dt = 0.3;                       // relaxation step factor
    double snap = 0.5;                     // interior nodes within snap x h of an interface join it
    int repair = 6;                        // max conformity repair rounds
    int smooth = 3;                        // guarded smoothing passes
    bool verbose = false;
};

struct Mesh2DStats {
    size_t nodes = 0, tris = 0, seeds = 0, junctions = 0;
    int iterations = 0, repair_rounds = 0;
    size_t repairs = 0, bad_edges = 0, spanning = 0;   // left after the repairs (0 = conforming)
    double min_angle = 0, q_min = 0, q_p5 = 0, q_median = 0;   // q = 4 sqrt(3) A / sum(l^2), 1 = equilateral
    std::vector<double> label_area, label_pixels;              // per label (mm^2)
    double ms_fields = 0, ms_relax = 0, ms_mesh = 0, ms_total = 0;
};

struct Mesh2D {
    std::vector<double> node;     // 2 per node, world coordinates
    std::vector<int32_t> tri;     // 3 per triangle, 0-based, counter-clockwise in pixel space
    std::vector<int32_t> label;   // per triangle
    // 4 per boundary edge: v0 v1 inner outer (outer = 0: the exterior), the inner
    // triangle on the left of v0 -> v1 (pixel space)
    std::vector<int32_t> edge;
};

// Set a 2-D option by name for the bindings (case-insensitive, '_' ignored):
// size hmin hmax k grad sigma graysigma nseed iters|maxiters fscale fsurf dt snap
// repair smooth verbose, thresholds (into `thr`), lsize ((label, size) pairs),
// isize ((a, b, size) triples, or `str` "h,L:h,A:B:h"; as set_option).
// Returns false for an unknown name.
bool set_option2d(Mesh2DOptions& o, std::vector<float>& thr, const std::string& name, const std::vector<double>& v,
                  const std::string& str = "");

// Mesh the image (labels, or im.gray with im.thresholds). Throws on bad input.
void mesh2d(Image2D& im, const Mesh2DOptions& o, Mesh2D& out, Mesh2DStats& st);

}  // namespace tn

#endif  // TRUSSNET_2D_H
