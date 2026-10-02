// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_modes.h -- the stages of the mesher on their own (--mode):
//   surface   a volume -> its region / exterior surfaces (the full mesher, faces out)
//   points    a volume -> the relaxed, labelled nodes (seeding + relaxation only)
//   check     a mesh or surface -> a report (quality; open / junction edges,
//             self-intersections of a surface)
//   tessellate  points -> their Delaunay tets (labelled from the node labels);
//             with v2mesh's labelled nodes (--mode points) and the image, the
//             mesher's full tessellation instead (v2m_pipeline start_nodes)
//   optimize  a labelled tet mesh -> the same mesh improved (the mesher's
//             optimiser: flips, kite removal, collapses, Steiner points, guarded
//             smoothing; region interfaces and the boundary are kept)
// and the helpers they share.

#ifndef V2MESH_MODES_H
#define V2MESH_MODES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "v2m_cdt.h"
#include "v2m_mesh.h"
#include "v2m_opt.h"
#include "v2m_pipeline.h"
#include "v2m_remesh.h"
#include "v2m_sdfshape.h"

namespace tn {

struct MeshReport {
    size_t nodes = 0, tets = 0, tris = 0;
    // tets
    size_t inverted = 0, flat = 0;     // negative / zero volume
    double min_dihedral = 0, joe_liu_min = 0, joe_liu_p5 = 0, joe_liu_med = 0;
    size_t slivers10 = 0;
    double volume = 0;
    double max_tet_volume = 0;         // the largest tet's
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

// --mode cdt on a surface (world): refuses a self-intersecting one; `fill` < 0 =
// default_cdt_fill; then the optimiser if o.opt (o.q, `opt_rounds`). Returns the
// fill spacing used.
double run_cdt(const Mesh& surf, const PipelineOptions& o, double fill, int opt_rounds, Mesh& out, CdtStats& cs,
               OptStats& os);
// --cdt-fill's default: o.grid.hbase if set, else 1.5 x the surface's mean edge
double default_cdt_fill(const Mesh& surf, double hbase);

// --mode remesh / repair: the surface's regions as a probability map meshed by
// their fields (o.tpm is set up for it: fields, no hole filling, channel 0 the
// exterior), into `lv`; `voxel` <= 0 = default_raster_voxel. Then run_pipeline(lv, o).
void remesh_volume(const Mesh& surf, double voxel, PipelineOptions& o, LabelVolume& lv, RasterStats& rs);
// Shape constructs (JSON, v2m_sdfshape.h) -> the labels' fields on a raster of
// `voxel` (0: --size / 3, else extent / 160) and their analytic program (lv.sdf)
void shapes_volume(const std::string& src, double voxel, bool clip, PipelineOptions& o, LabelVolume& lv,
                   ShapeScene& sc);
// --raster-voxel's default: the smaller of hbase / 3 (else extent / 160) and half
// the surface's mean edge (and no finer than extent / 600)
double default_raster_voxel(const Mesh& surf, double hbase);

// the region surfaces of labelled tets, as MeshTri (+ inner / outer labels), into m.tris
void add_faces(Mesh& m);

// Drop the nodes no element refers to (renumbering tets, tris and the node
// attributes).
void compact_nodes(Mesh& m);

}  // namespace tn

#endif  // V2MESH_MODES_H
