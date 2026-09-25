// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// tn_volume.h -- a multi-label volume: a discrete uint16 label grid, x-fastest
// (data[(z*ny + y)*nx + x]), with its voxel size and the 4x4 voxel->world affine.
// Label 0 is the EXTERIOR: it never holds mesh nodes and no tet may lie in it.
// Loaded from .nii[.gz] / .jnii / .bnii through the siamize readers (src/io),
// reoriented to canonical RAS.

#ifndef TRUSSNET_LABELVOL_H
#define TRUSSNET_LABELVOL_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tn {

struct LabelVolume {
    std::vector<uint16_t>  data;             // x-fastest: data[(z*ny + y)*nx + x]
    int                    nx = 0, ny = 0, nz = 0;
    std::array<double, 3>  voxelsize{ { 1, 1, 1 } };   // (x,y,z) mm
    std::array<double, 16> affine{};         // 4x4 row-major, voxel(x,y,z)->world(mm)
    int                    maxlabel = 0;     // highest label value present
    // gray-scale input (optional): the intensity (same layout as data) and the
    // ascending thresholds; label = number of thresholds <= intensity (0 = below
    // the first = exterior). The interfaces are then the iso-surfaces I = t_k.
    std::vector<float>     gray;
    std::vector<float>     thresholds;

    int64_t numel() const {
        return static_cast<int64_t>(data.size());
    }
};

// Load `path` into a label grid (keep_gray: also keep the float intensity in
// `gray`, for apply_thresholds). Throws std::runtime_error on I/O errors.
LabelVolume load_label_volume(const std::string& path, bool keep_gray = false);

// Gray-scale mode: optionally smooth `gray` (Gaussian, sigma voxels, 0 = none),
// store the ascending thresholds and relabel data from them.
void apply_thresholds(LabelVolume& lv, const std::vector<float>& thresholds, float sigma = 0.0f);

}  // namespace tn

#endif  // TRUSSNET_LABELVOL_H
