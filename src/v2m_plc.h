// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_plc.h -- TetGen piecewise linear complexes (.poly, .smesh; the points in a
// .node file beside them when the .poly lists none) -> a triangle surface.
//
//   * a facet (one or more polygons, possibly with holes, in one plane) comes
//     out as triangles: a constrained triangulation of its polygons' edges, less
//     what lies outside them and round its hole points (a polygon inside another
//     without a hole in it stays part of the facet);
//   * a polygon of two corners (a segment) or one (a point) inside a facet is a
//     vertex / edge of that triangulation; a facet with no area (segments or
//     points only) has no triangles and is counted, not kept -- the CDT keeps
//     faces, not loose edges;
//   * the volume holes (part 3) and region attributes (part 4) come back as
//     seeds (Mesh::seed_holes / seed_regions) for --mode cdt: the compartment
//     holding a hole point is dropped, one holding a region point gets its
//     region number as the label.
//
// Facet boundary markers and the regions' volume constraints are read and not
// used. Indices may start at 0 or 1 (the first point's decides, as in TetGen).

#ifndef V2MESH_PLC_H
#define V2MESH_PLC_H

#include <cstddef>
#include <string>
#include <vector>

#include "v2m_mesh.h"

namespace tn {

struct PlcStats {
    size_t points = 0, facets = 0, polygons = 0, facet_holes = 0;
    size_t triangles = 0;
    size_t flat_facets = 0;   // facets with no area (segments / points only): dropped
    size_t holes = 0, regions = 0;
};

// Read a .poly / .smesh file. Throws on a malformed file or a facet whose
// polygon edges cross.
Mesh read_poly(const std::string& path, PlcStats* st = nullptr);

// A constrained triangulation of 2-D points P (2 per point, distinct) keeping
// the segments segs (2 point indices each; a point on a segment splits it):
// triangles (3 per, ccw) over the points' convex hull, each with its component
// comp[t] (the triangles connected across non-constraint edges), comp_out[c]
// set when component c reaches the hull. Throws when two segments cross.
void cdt2d(const std::vector<double>& P, const std::vector<int>& segs, std::vector<int>& tris, std::vector<int>& comp,
           std::vector<char>& comp_out);

}  // namespace tn

#endif  // V2MESH_PLC_H
