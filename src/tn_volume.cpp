// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// tn_volume.cpp -- see tn_volume.h.

#include "tn_volume.h"

#include <cmath>
#include <stdexcept>

#include "nifti_io.h"
#include "jnifti_io.h"
#include "siam.h"

namespace tn {

namespace {

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

}  // namespace

LabelVolume load_label_volume(const std::string& path) {
    // Dispatch by extension. siamize returns a canonical (Z,Y,X) float Volume
    // plus affine; 4D inputs are rejected by its reader (TPM argmax TODO).
    siam::NiftiImage img;

    if (ends_with(path, ".jnii") || ends_with(path, ".bnii")) {
        img = siam::load_jnifti_ras(path);
    } else if (ends_with(path, ".nii") || ends_with(path, ".nii.gz")) {
        img = siam::load_nifti_ras(path);
    } else {
        throw std::runtime_error("unsupported extension (want .nii/.nii.gz/.jnii/.bnii): " + path);
    }

    const int64_t Z = img.volume.shape[0];
    const int64_t Y = img.volume.shape[1];
    const int64_t X = img.volume.shape[2];
    const int64_t n = Z * Y * X;

    if (n <= 0 || static_cast<int64_t>(img.volume.data.size()) != n) {
        throw std::runtime_error("empty or malformed volume: " + path);
    }

    LabelVolume lv;
    lv.nx = static_cast<int>(X);
    lv.ny = static_cast<int>(Y);
    lv.nz = static_cast<int>(Z);
    lv.voxelsize = { { img.zooms_canon[0], img.zooms_canon[1], img.zooms_canon[2] } };

    for (int i = 0; i < 16; ++i) {
        lv.affine[i] = static_cast<double>(img.affine_canon[i]);
    }

    // The canonical Volume is already x-fastest (data[(z*Y+y)*X + x]); round
    // float -> integer label (0 = exterior).
    lv.data.resize(static_cast<size_t>(n));
    int maxlabel = 0;

    for (int64_t i = 0; i < n; ++i) {
        long v = std::lround(img.volume.data[static_cast<size_t>(i)]);

        if (v < 0) {
            v = 0;
        }

        if (v > 65534) {
            v = 65534;
        }

        lv.data[static_cast<size_t>(i)] = static_cast<uint16_t>(v);

        if (static_cast<int>(v) > maxlabel) {
            maxlabel = static_cast<int>(v);
        }
    }

    lv.maxlabel = maxlabel;
    return lv;
}

}  // namespace tn
