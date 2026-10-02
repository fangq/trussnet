// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_meshio.h -- reading meshes and point clouds, for the modes that start from
// one (--mode tessellate / optimize / cdt / remesh / check):
//
//   .jmsh / .bmsh  JMesh (text JSON / binary BJData): MeshNode (N x 3, or N x 4
//                  with a node label), MeshElem (M x 4 or M x 5 with a label),
//                  MeshTri / MeshSurf (M x 3 triangles, + 1 label or + an
//                  inner / outer label pair), NodeLabel / NodeType /
//                  NodePartner (v2mesh's --mode points). Arrays may be plain
//                  (nested) JSON arrays or JData-annotated, zlib-compressed or
//                  not. Indices are 1-based in the file.
//   .off           triangles (polygons are fanned)
//   .stl           ASCII or binary; coincident corners are welded
//   .xyz / .txt    points, one per line: x y z [label]
//
// Everything comes back 0-based in a tn::Mesh (the writer's container).

#ifndef V2MESH_MESHIO_H
#define V2MESH_MESHIO_H

#include <string>

#include "v2m_mesh.h"

namespace tn {

// Read `path` (format by extension; see above). Throws std::runtime_error.
// Triangles with a single label column get tri_labels (label, 0); a node label
// column (MeshNode N x 4, .xyz 4th column) fills node_labels.
Mesh read_mesh(const std::string& path);

// The tessellation of a STEP file read_mesh reads (.step / .stp; v2m_step.h):
// chord tolerance and max edge length (mm, 0 = automatic / none), max angle
// (degrees).
void set_step_options(double tol, double angle, double size);

// Shape constructs read_mesh reads (a .json of MCX / JMesh shapes: their exact
// surface, v2m_csgsurf.h): --shape-clip, --overlap.
void set_shape_options(bool clip, const std::string& overlap);

}  // namespace tn

#endif  // V2MESH_MESHIO_H
