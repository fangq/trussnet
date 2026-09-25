// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_tetra.h -- stage 10: exact Delaunay of the relaxed nodes (vendored Diazzi
// et al. code), restricted-Delaunay labelling by circumcentre (tets whose
// circumcentre lies in label 0 are removed -- that takes out every hull and
// concavity tet), the hard conformity checks, and quality statistics.

#ifndef TRUSSNET_TETRA_H
#define TRUSSNET_TETRA_H

#include <cstdint>
#include <vector>

#include "tn_grid.h"
#include "tn_particles.h"

namespace tn {

struct TetOut {
    std::vector<float> P;          // nodes (grid mm), = Nodes::P
    std::vector<int32_t> tets;     // 4 per tet, kept tets only
    std::vector<int32_t> label;    // per tet
};

struct TetStats {
    size_t delaunay_tets = 0, kept = 0, peeled = 0;
    int repair_rounds = 0;
    size_t repaired = 0;
    // conformity: (a) a boundary / interface face with a node not on that
    // interface, (b) a kept edge crossing label 0, (c) an interior node inside a
    // tet of another label
    size_t bad_faces = 0, bad_edges = 0, bad_span = 0;
    // quality over kept tets
    double min_dihedral = 0, joe_liu_min = 0, joe_liu_p5 = 0, joe_liu_med = 0;
    size_t slivers10 = 0, slivers5 = 0;
    double volume = 0;
    double ms_delaunay = 0, ms_label = 0, ms_check = 0;
};

// Tessellate, then repair (up to max_repair rounds): the crossing tets and the
// edges through label 0 get interface nodes at their crossings (restricted-
// Delaunay refinement) and the mesh is rebuilt. `nd` gains / moves those nodes.
void tessellate(const Grid& g, Nodes& nd, bool voxel_mode, int max_repair, TetOut& m, TetStats& st);

}  // namespace tn

#endif  // TRUSSNET_TETRA_H
