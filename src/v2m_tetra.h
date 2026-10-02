// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_tetra.h -- stage 10: exact Delaunay of the relaxed nodes (vendored Diazzi
// et al. code), restricted-Delaunay labelling by circumcentre (tets whose
// circumcentre lies in label 0 are removed -- that takes out every hull and
// concavity tet), the hard conformity checks, and quality statistics.

#ifndef V2MESH_TETRA_H
#define V2MESH_TETRA_H

#include <cstdint>
#include <string>
#include <vector>

#include "v2m_grid.h"
#include "v2m_particles.h"

namespace tn {

struct TetOut {
    std::vector<float> P;          // nodes (grid mm), = Nodes::P
    // --mode surface with --surf-smooth: the nodes before the smoothing -- its tets
    // are only the surfaces' scaffold and may turn over, so the faces are oriented
    // by these (extract_mesh_faces); empty otherwise
    std::vector<float> P_orient;
    std::vector<int32_t> tets;     // 4 per tet, kept tets only
    std::vector<int32_t> label;    // per tet
};

// Volume-preserving smoothing of the region surfaces after the tessellation, as
// iso2mesh's smoothsurf: 'laplacianhc' (Vollmer's HC: a Laplacian step, then the
// difference from the original pushed back), 'lowpass' (Taubin: +alpha, then
// -1.02 alpha) or 'laplacian'. A node on one interface moves with its neighbours
// on it; one on a junction curve (where 3+ regions meet) along the curve; corners
// stay. A move that would invert a tet, or turn an interface triangle over, is not
// made. iters = 0: off.
struct SurfSmoothParams {
    int iters = 0;
    std::string method = "laplacianhc";
    double alpha = 0.5, beta = 0.5;
};

struct TetStats {
    size_t delaunay_tets = 0, kept = 0, peeled = 0;
    int repair_rounds = 0;
    size_t repaired = 0;
    // conformity: (a) a boundary / interface face with a node not on that
    // interface, (b) a kept edge crossing label 0, (c) an interior node inside a
    // tet of another label
    size_t bad_faces = 0, bad_edges = 0, bad_span = 0;
    bool canonical = true;   // the last tessellation's tets in canonical order (a fresh Delaunay)
    // how far off: distance (voxels) of each offending node from the interface it
    // should lie on (smoothed field, |psi| / |grad psi|); p50 / p95 / p99 / max
    double dev_face[4] = { 0, 0, 0, 0 }, dev_span[4] = { 0, 0, 0, 0 };
    std::vector<double> label_vol, label_vox;   // per label: mesh volume, voxel volume (mm^3)
    // quality over kept tets
    double min_dihedral = 0, joe_liu_min = 0, joe_liu_p5 = 0, joe_liu_med = 0;
    size_t slivers10 = 0, slivers5 = 0;
    size_t sliver_by_interior[5] = { 0, 0, 0, 0, 0 };   // slivers (< 10 deg) by # interior nodes
    double volume = 0;
    double ms_delaunay = 0, ms_label = 0, ms_check = 0, ms_smooth = 0;
    size_t smoothed = 0;
    size_t surf_moved = 0, surf_nodes = 0, surf_blocked = 0;   // --surf-smooth: node moves, nodes, moves undone
    double ms_surf = 0;
    size_t presnapped = 0;   // interior nodes put on an interface before the Delaunay
    size_t coincident = 0;   // relaxed nodes dropped for sitting exactly on another
    size_t q_added = 0;   // nodes added by the radius-edge (-q) refinement
    int q_rolled_back = 0;   // 1 if a refinement round was undone (it cost conformity)
    int opt_flips32 = 0, opt_flips23 = 0, opt_collapses = 0, opt_steiner = 0, opt_moves = 0, opt_kites = 0;
    double ms_opt = 0;   // accepted interior-node moves of the ODT smoothing
};

// Tessellate, then repair (up to max_repair rounds): the crossing tets and the
// edges through label 0 get interface nodes at their crossings (restricted-
// Delaunay refinement) and the mesh is rebuilt. `nd` gains / moves those nodes.
// Run the full Delaunay builds on OpenCL device `device` (-1 = first GPU; -2, the
// default, = the exact CPU code). The result is identical either way.
void set_gpu_delaunay(int device);

// Quality of one tet: minimum dihedral angle (degrees), Joe-Liu quality (0..1)
// and signed volume.
void tet_quality(const double* p[4], double& mindih, double& jl, double& vol);

// surface_only: `nd` holds only the surface nodes (--mode surface): the sculpting
// peels only the tets that lie outside, not the flat ones.
void tessellate(const Grid& g, Nodes& nd, bool voxel_mode, int max_repair, TetOut& m, TetStats& st, int smooth = 5,
                bool opt = true, double q = 2.0, bool surface_only = false, const SurfSmoothParams* ss = nullptr);

}  // namespace tn

#endif  // V2MESH_TETRA_H
