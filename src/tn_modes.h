// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_modes.h -- the stages of the mesher on their own (--mode):
//   surface   a volume -> its region / exterior surfaces (the full mesher, faces out)
//   points    a volume -> the relaxed, labelled nodes (seeding + relaxation only)
//   check     a mesh or surface -> a report (quality; open / junction edges,
//             self-intersections of a surface)
//   tessellate  points -> their Delaunay tets (labelled from the node labels);
//             with trussnet's labelled nodes (--mode points) and the image, the
//             mesher's full tessellation instead (tn_pipeline start_nodes)
//   optimize  a labelled tet mesh -> the same mesh improved (the mesher's
//             optimiser: flips, kite removal, collapses, Steiner points, guarded
//             smoothing; region interfaces and the boundary are kept)
// and the helpers they share.

#ifndef TRUSSNET_MODES_H
#define TRUSSNET_MODES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tn_mesh.h"
#include "tn_opt.h"

namespace tn {

struct MeshReport {
    size_t nodes = 0, tets = 0, tris = 0;
    // tets
    size_t inverted = 0, flat = 0;     // negative / zero volume
    double min_dihedral = 0, joe_liu_min = 0, joe_liu_p5 = 0, joe_liu_med = 0;
    size_t slivers10 = 0;
    double volume = 0;
    std::vector<double> label_vol;     // per tet label
    // surfaces (a tet mesh: its region surfaces)
    size_t open_edges = 0;             // on one triangle
    size_t junction_edges = 0;         // on > 2 (allowed where regions meet)
    size_t region_open_edges = 0;      // a region's boundary not closed (per region, summed)
    size_t self_intersections = 0;     // pairs of triangles that cross (not counting shared corners / edges)
    std::vector<std::pair<int32_t, int32_t>> crossing;   // the first few pairs
    bool ok() const {
        return inverted == 0 && open_edges == 0 && region_open_edges == 0 && self_intersections == 0;
    }
};

// Report on `m` (tets and / or triangles). Surfaces: triangles in `m.tris`
// (tri_labels = inner / outer pairs, or none); a tet mesh is checked on its
// region surfaces too (extract_faces).
MeshReport check_mesh(const Mesh& m);
void print_report(const MeshReport& r, const std::string& name);

// Pairs of triangles that cross: exact orientation tests (the vendored
// predicates) on the candidates of a uniform grid; pairs sharing an edge count
// only if they fold onto each other, pairs sharing a corner only where their
// other edges cross. Returns the count; `pairs` (optional) the first `keep`.
size_t self_intersections(const std::vector<double>& nodes, const std::vector<int32_t>& tris,
                          std::vector<std::pair<int32_t, int32_t>>* pairs = nullptr, size_t keep = 20);

// Optimise the labelled tets of `m` in place (world coordinates). The node types
// the optimiser needs come from the tet labels: a node's label set is the labels
// of its tets, plus the exterior (0) on the boundary; nodes on an interface or
// the boundary are frozen, so the regions keep their surfaces.
void optimize_tets(Mesh& m, const OptParams& prm, OptStats& os);

// The Delaunay tets of m.nodes (the convex hull filled), in place; each tet
// takes the most frequent label of its nodes (ties: the smallest), label 1 when
// the nodes have none. `gpu` > -2: the OpenCL Delaunay on that device if it can.
void tessellate_points(Mesh& m, int gpu);

// Drop the nodes no element refers to (renumbering tets, tris and the node
// attributes).
void compact_nodes(Mesh& m);

}  // namespace tn

#endif  // TRUSSNET_MODES_H
