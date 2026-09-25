// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_grid.h -- stage 2: the brick-sparse smooth interface fields phi_l, the
// curvature-driven, gradient-limited sizing field h and its 256 log grades
// (see src/opencl/tn_grid_body.cl for the per-voxel definitions).

#ifndef TRUSSNET_GRID_H
#define TRUSSNET_GRID_H

#include <cstdint>
#include <vector>

#include "tn_volume.h"

namespace tn {

struct GridParams {
    float sigma = 1.0f;     // Gaussian smoothing of the label indicators (voxels): the
    //                         smooth interfaces the particles are trapped on
    float thick = 0.0f;     // thin layers: h <= thickness / thick (0 = off)
    float thin_floor = 0.5f; // smallest thin-layer size (voxels)
    float preserve = 0.0f;  // label-preserving margin at voxel centres (0 = off)
    float sigma_curv = 2.0f; // wider smoothing used only to estimate curvature
    float hbase = 0.0f;     // default element size (mm); 0 = 3 x the smallest voxel side
    float hmin = 0.0f;      // smallest size (mm); 0 = hbase / 3
    float hmax = 0.0f;      // largest size (mm); 0 = hbase
    float K = 3.0f;         // elements per radian of curvature: h <= 1/(K |kappa|)
    float g = 0.3f;         // gradient limit |grad h| <= g
    std::vector<float> hlab; // optional per-label size (mm), index = label; 0 = hbase
};

struct Grid {
    // dimensions
    int nx = 0, ny = 0, nz = 0;       // voxels
    int nbx = 0, nby = 0, nbz = 0;    // 8^3 bricks
    float vs[3] = { 1, 1, 1 };        // voxel size (mm)
    int R = 3;                        // brick label radius (voxels): max of both smoothings
    float sigma = 1.0f;
    const std::vector<uint16_t>* L = nullptr;   // the labels (owned by the LabelVolume)
    int nlab = 0;                     // maxlabel + 1

    // bricks: labels within R (ascending, TN_BL slots, 0xFFFF padded), their count
    // (TN_BL+1 = overflow) and the first slot (-1 = uniform)
    std::vector<int> bl_cnt;
    std::vector<uint16_t> bl_lab;
    std::vector<int> bl_slot;
    // slots: (brick, label) and 8^3 phi values each
    std::vector<int> slot_brick;
    std::vector<uint16_t> slot_label;
    std::vector<float> phi;

    // sizing
    float hmin = 0, hmax = 0, hbase = 0;
    std::vector<float> h;             // per voxel (mm)
    std::vector<uint8_t> grade;       // per voxel, 256 log grades in [hmin, hmax]
    int limit_sweeps = 0;             // gradient-limiting sweeps run
    int overflow_bricks = 0;          // bricks with more than TN_BL labels nearby
};

// Build the grid on the host (OpenMP; the reference for the OpenCL path).
void build_grid_cpu(const LabelVolume& lv, const GridParams& prm, Grid& g);

}  // namespace tn

#endif  // TRUSSNET_GRID_H
