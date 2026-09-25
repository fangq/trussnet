// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_pipeline.h -- the whole mesher as one call (volume -> sizing grid -> seeds ->
// relaxation -> tessellation / repair / optimisation), shared by the command-line
// driver, the MATLAB/Octave MEX (tn_mex.cpp) and the Python module (pytrussnet.cpp).

#ifndef TRUSSNET_PIPELINE_H
#define TRUSSNET_PIPELINE_H

#include <cstdint>
#include <string>
#include <vector>

#include "tn_grid.h"
#include "tn_particles.h"
#include "tn_tetra.h"
#include "tn_volume.h"

namespace tn {

struct PipelineOptions {
    GridParams grid;
    RelaxParams relax;
    int max_repair = 6;
    int smooth = 5;
    bool opt = true;
    double q = 2.0;                   // radius-edge bound (-q); 0 = off
    int gpu = -2;                     // -2: CPU (OpenMP); else the OpenCL device (-1 = first GPU)
    std::vector<float> thresholds;    // gray-scale input: iso-values (needs lv.gray)
    float gray_sigma = 0.0f;
    bool report = false;              // print the per-stage summary lines
    std::string dump_grid, dump_nodes;   // debug dumps (CLI)
};

struct PipelineResult {
    TetOut mesh;                      // nodes in grid mm (voxel i at i * voxelsize), 0-based tets, labels
    TetStats tess;
    RelaxStats relax;
    size_t seeds = 0;
    bool used_gpu = false;
    double ms_input = 0, ms_grid = 0, ms_seed = 0, ms_relax = 0, ms_tess = 0, ms_total = 0;
};

// Set one option by name for the bindings (MATLAB struct fields / Python keyword
// arguments). Names are case-insensitive and ignore '_' (sigma_thin = sigmathin):
//   size hmin hmax k grad sigma sigmathin thick thinfloor preserve     (sizing, mm)
//   nseed iters|maxiters fscale fsurf dt snap jseed corners trap       (relaxation)
//   q|reratio|quality opt smooth repair                                 (tessellation)
//   thresholds graysigma gpu gpuid verbose
//   lsize: (label, size) pairs, flattened
// `v` holds the numeric value(s), `str` a string value (trap). Returns false for
// an unknown name.
bool set_option(PipelineOptions& o, const std::string& name, const std::vector<double>& v,
                const std::string& str = "");

// Mesh `lv` (labels, or gray-scale when o.thresholds is set: lv.gray is then
// relabelled in place). If o.gpu > -2 and OpenCL fails, falls back to the CPU.
void run_pipeline(LabelVolume& lv, const PipelineOptions& o, PipelineResult& r);

// the nodes in world coordinates: world = affine * [P / voxelsize; 1]
void nodes_to_world(const LabelVolume& lv, const TetOut& m, std::vector<double>& world);

// Boundary faces of the labelled tets: 5 ints per face (v0, v1, v2, inner label,
// outer label; outer = 0 on the exterior surface), 0-based, oriented with the
// normal pointing from the inner label to the outer one. Each interface once.
void extract_faces(const std::vector<int32_t>& tets, const std::vector<int32_t>& labels,
                   const std::vector<double>& nodes, std::vector<int32_t>& faces);

}  // namespace tn

#endif  // TRUSSNET_PIPELINE_H
