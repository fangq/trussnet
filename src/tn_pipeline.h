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
#include "tn_tpm.h"
#include "tn_manifold.h"
#include "tn_surflabel.h"
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
    TpmOptions tpm;                   // 4-D (tissue-probability) input: see tn_tpm.h
    float gray_sigma = 0.0f;
    bool report = false;              // print the per-stage summary lines
    std::string dump_grid, dump_nodes;   // debug dumps (CLI)
    bool stop_after_relax = false;       // --mode points: seeding + relaxation only (r.nodes)
    const Nodes* start_nodes = nullptr;  // --mode tessellate: these nodes (grid mm) instead of seeding + relaxing
    // --mode surface / repair: only the surfaces are wanted. The interior nodes are
    // dropped before the tessellation (they never lie on a surface) and the
    // quality stages (-q, ODT, the optimiser: interior work) are skipped -- about
    // half the time, and on Colin27 slightly more accurate surfaces
    bool surface_only = false;
    SurfLabelOptions surf;               // --mode cdt / remesh / repair: regions of surfaces (tn_surflabel.h)
    // --manifold: no pinched edges in the region surfaces (tn_manifold.h); nest:
    // labels outermost first (CSF, GM, WM: the inner one is joined at a pinch)
    bool manifold = false;
    std::vector<int> nest;
    // shape input (JSON, tn_sdfshape.h): the raster spacing (0: size / 3, else
    // extent / 160) and whether the objects are cut to the first
    double shape_voxel = 0.0;
    bool shape_clip = true;
};

struct PipelineResult {
    TetOut mesh;                      // nodes in grid mm (voxel i at i * voxelsize), 0-based tets, labels
    TetStats tess;
    RelaxStats relax;
    size_t seeds = 0;
    size_t thinned = 0;               // nodes removed by --thin
    bool used_gpu = false;
    Nodes nodes;                      // the relaxed nodes (stop_after_relax; grid mm)
    ManifoldStats manifold;           // --manifold
    double ms_input = 0, ms_grid = 0, ms_seed = 0, ms_relax = 0, ms_tess = 0, ms_total = 0;
};

// Set one option by name for the bindings (MATLAB struct fields / Python keyword
// arguments). Names are case-insensitive and ignore '_' (sigma_thin = sigmathin):
//   size hmin hmax k grad sigma sigmathin thick thinfloor preserve     (sizing, mm)
//   nseed iters|maxiters fscale fsurf dt snap jseed corners trap thin  (relaxation)
//   q|reratio|quality opt smooth repair                                 (tessellation)
//   thresholds graysigma gpu gpuid verbose
//   tpmexterior (0-based channels) tpmmap tpmspm6 tpmsigma tpmholes tpmfields  (4-D input)
//   tpmthresh: one threshold (every tissue label) or (label, threshold) pairs, or
//              `str` "T,L:T,..."
//   lsize: (label, size) pairs, flattened
//   isize: interface sizes, (a, b, size) triples flattened (a = b = -1: every
//          interface; b = -1: every interface of label a), or `str` "h,L:h,A:B:h"
// `v` holds the numeric value(s), `str` a string value (trap). Returns false for
// an unknown name.
bool set_option(PipelineOptions& o, const std::string& name, const std::vector<double>& v,
                const std::string& str = "");

// Mesh `lv` (labels, or gray-scale when o.thresholds is set: lv.gray is then
// relabelled in place). If o.gpu > -2 and OpenCL fails, falls back to the CPU.
void run_pipeline(LabelVolume& lv, const PipelineOptions& o, PipelineResult& r);

// Load a volume file as the CLI does: a 4-D .jnii/.bnii/.nii[.gz] is a TPM (o.tpm),
// else labels, or a gray-scale intensity when o.thresholds is set. `tpm_filled`
// (optional) receives the enclosed exterior voxels filled (TPM).
LabelVolume load_volume_file(const std::string& path, const PipelineOptions& o, size_t* tpm_filled = nullptr,
                             std::vector<int>* tpm_map = nullptr);

// A user sizing (the bindings' `sizing` option), 0 = automatic, as either
//   - one value per voxel of lv (x fastest): the sizing field (o.grid.hvox), or
//   - one per TPM channel (tpm_map = the channel -> label map of apply_tpm), or
//   - one per label: labels 1..N (N values) or 0..N (N + 1; entry 0 ignored);
//     N = the thresholds' count for a gray-scale volume.
// Per-label / per-channel sizes go to o.grid.hlab (a channel sharing a label:
// the smallest). Throws on any other length.
void apply_user_sizing(const LabelVolume& lv, const std::vector<double>& h, const std::vector<int>& tpm_map,
                       PipelineOptions& o);

// the nodes in world coordinates: world = affine * [P / voxelsize; 1]
void nodes_to_world(const LabelVolume& lv, const TetOut& m, std::vector<double>& world);

// the inverse: world coordinates -> grid mm (the nodes' frame)
void world_to_nodes(const LabelVolume& lv, const std::vector<double>& world, std::vector<float>& P);

// Boundary faces of the labelled tets: 5 ints per face (v0, v1, v2, inner label,
// outer label; outer = 0 on the exterior surface), 0-based, oriented with the
// normal pointing from the inner label to the outer one. Each interface once.
void extract_faces(const std::vector<int32_t>& tets, const std::vector<int32_t>& labels,
                   const std::vector<double>& nodes, std::vector<int32_t>& faces);

}  // namespace tn

#endif  // TRUSSNET_PIPELINE_H
