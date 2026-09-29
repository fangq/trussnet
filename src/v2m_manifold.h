// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_manifold.h -- region surfaces without pinched edges (--manifold).
//
// A pinched edge lies on 4+ faces between the same two labels a | b: going
// round it, a b a b -- two parts of a touch only along the edge, as do two of
// b. It reproduces an ambiguity of the input (voxels of one label touching only
// diagonally, a "checkerboard"; the smoothed field has a saddle on the
// interface there), and makes the region surfaces non-manifold. Each is opened
// one way, as marching cubes does its ambiguous cases, by relabelling the tets
// of one wedge round the edge (no node moves):
//
//   * with the exterior (a = 0): the smaller wedge of b is removed (no exterior
//     tets exist to fill in);
//   * `nest` (labels, outermost first -- e.g. CSF, GM, WM): the inner label is
//     joined -- the outer layer is locally of zero thickness, so the smaller
//     wedge of the outer label takes the inner one (two GM banks meet across a
//     vanished CSF sheet);
//   * else: the smallest wedge takes the other label.
//
// Repeated until no edge is pinched (relabelling can pinch a neighbour). With
// `nest`, a piece of an outer label that the relabelling cut off, enclosed by
// inner labels only (a CSF pocket inside GM), is merged into its enclosing label:
// the outer label stays topologically outside. Pieces that were separate before
// (ventricles inside WM) are left alone.

#ifndef V2MESH_MANIFOLD_H
#define V2MESH_MANIFOLD_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tn {

struct ManifoldStats {
    size_t pinched_before = 0, pinched_after = 0;
    size_t relabelled = 0, removed = 0;   // tets given another label / the exterior
    size_t pockets = 0, pocket_tets = 0;  // cut-off pieces of an outer label merged
    int rounds = 0;
    double ms = 0;
};

// tets: 4 per tet (0-based), labels: per tet (> 0); X: 3 per node. Tets that
// become exterior are removed from both arrays.
template <typename T>
void make_manifold(std::vector<int32_t>& tets, std::vector<int32_t>& labels, const std::vector<T>& X,
                   const std::vector<int>& nest, ManifoldStats& st);

// The pinched edges of a labelled tet mesh (for reports and tests).
size_t count_pinched_edges(const std::vector<int32_t>& tets, const std::vector<int32_t>& labels);

}  // namespace tn

#endif  // V2MESH_MANIFOLD_H
