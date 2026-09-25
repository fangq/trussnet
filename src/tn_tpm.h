// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_tpm.h -- tissue-probability-map (TPM) input: a 4-D volume of C per-voxel
// class probabilities (.jnii / .bnii / .nii[.gz], e.g. SPM's 6 classes or
// siamize's 18). Each mesh label's interface field is its own probability, so
// the a|b interface is exactly p_a = p_b (sub-voxel, no smoothing of a hard
// segmentation), and the labels are the argmax. Adapted from gpu_brain2mesh's
// b2m_tpm (same author), generalized to any channel count and label map.

#ifndef TRUSSNET_TPM_H
#define TRUSSNET_TPM_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "tn_volume.h"

namespace tn {

// the raw channels, channel-major: p[c * nv + (z * ny + y) * nx + x]
struct Tpm {
    std::vector<float> p;
    int C = 0, nx = 0, ny = 0, nz = 0;
    std::array<double, 3> voxelsize{ { 1, 1, 1 } };
    std::array<double, 16> affine{ { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } };   // row-major, voxel -> world
    std::vector<std::string> names;   // channel names (LabelTable), may be empty

    size_t nv() const {
        return static_cast<size_t>(nx) * ny * nz;
    }
};

struct TpmOptions {
    // channel -> mesh label (label 0 = exterior; several channels may share a
    // label: their probabilities add). Empty: the exterior channels -> 0, every
    // other channel its own label 1, 2, ... in channel order.
    std::vector<int> map;
    // the exterior channels (0-based) for the default map; empty: the channels
    // named background / air / bg / outside / exterior / none
    std::vector<int> exterior;
    bool spm6 = false;       // merge siamize's 18 classes to SPM6 (GM WM CSF Bone Soft, Air = exterior)
    float sigma = 0.0f;      // Gaussian smoothing of the probabilities (voxels), 0 = none
    // fill the exterior pockets not connected to the volume boundary (sinuses,
    // airways enclosed by tissue; brain2mesh's fill-holes): they take the nearest
    // tissue label, probability 1; and no exterior probability deeper than 2 voxels
    // inside the tissue (the tissues renormalized there)
    bool fill_holes = true;
    // interfaces from the probabilities themselves (smoothed p_a = p_b) instead of
    // the default: the argmax labels meshed like a label volume. On the tested TPMs
    // (a network's near-binary softmax; the ANTS atlas, binary at the head surface)
    // the default was as accurate and more robust (fewer residual crossings / slivers)
    bool fields = false;
};

// Read a 4-D TPM (.jnii / .bnii / .nii / .nii.gz). Throws if the file is not 4-D.
Tpm load_tpm(const std::string& path);

// True if `path` holds a 4-D volume (cheap for .nii[.gz] and text .jnii; parses a .bnii).
bool is_tpm_file(const std::string& path);

// Fill `lv` from the TPM: lv.data = the argmax of the per-label probabilities
// (label 0 = exterior, 1 - sum(tissue) when no channel maps to 0), maxlabel, voxel
// size, affine and lv.soft_volume; with o.fields also lv.prob (label-major). Returns the channel -> label map used;
// `filled` (optional) receives the number of exterior voxels filled.
std::vector<int> apply_tpm(const Tpm& t, const TpmOptions& o, LabelVolume& lv, size_t* filled = nullptr);

}  // namespace tn

#endif  // TRUSSNET_TPM_H
