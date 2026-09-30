// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_step.h -- CAD models in STEP (ISO 10303-21: AP203 / AP214 / AP242 B-reps)
// -> a closed, labelled triangle surface, for --mode cdt (the faces kept) or
// remesh / repair.
//
//   * the file: its DATA section parsed into entity instances (simple and
//     complex), the length unit (SI prefixes, inch / foot ..) scaled to mm and
//     the plane angle unit (radian / degree) found;
//   * the solids: MANIFOLD_SOLID_BREP and BREP_WITH_VOIDS (a void shell
//     labelled like its solid: a hole), else the shells of a
//     SHELL_BASED_SURFACE_MODEL; one label per solid, 1, 2, .. in file order;
//   * the geometry: lines, circles, ellipses, (rational) B-spline curves,
//     polylines, surface / seam / trimmed curves; planes, cylinders, cones,
//     spheres, tori, (rational) B-spline surfaces, surfaces of revolution and
//     of linear extrusion;
//   * the tessellation: every edge sampled once (a chord tolerance and 15
//     degrees a segment) and shared by its two faces, so the surface is
//     watertight; each face triangulated in its parameter plane -- its loops
//     unwrapped on a periodic surface, cut along a seam where they go round
//     it, closed at a pole or apex -- with interior points on a curved one,
//     by a constrained triangulation (v2m_plc.h cdt2d), mapped back.
//
// Not read: assemblies' placed instances (MAPPED_ITEM, CONTEXT_DEPENDENT_SHAPE_
// REPRESENTATION transforms: each solid is taken where it is defined), offset
// surfaces and curves, and faceted / wireframe representations; a face that
// cannot be tessellated is counted and skipped (the surface then has a hole).

#ifndef V2MESH_STEP_H
#define V2MESH_STEP_H

#include <cstddef>
#include <string>
#include <vector>

#include "v2m_mesh.h"

namespace tn {

struct StepOptions {
    double tol = 0;          // chord tolerance (mm); 0 = 0.05 % of the model's size
    double angle = 15;       // max angle (degrees) a segment / triangle turns
    double size = 0;         // max edge length (mm; --size; 0 = the curvature alone)
};

struct StepStats {
    size_t entities = 0, solids = 0, shells = 0, faces = 0, edges = 0;
    size_t faces_failed = 0, triangles = 0, nodes = 0;
    double unit = 1;          // file length unit in mm
    double tol = 0;           // the chord tolerance used (mm)
    std::vector<std::string> notes;   // unsupported entities / failed faces (first few)
};

// Read a .step / .stp file (world coordinates in mm). Throws on a malformed
// file or one with no B-rep solid or shell.
Mesh read_step(const std::string& path, const StepOptions& o = StepOptions(), StepStats* st = nullptr);

}  // namespace tn

#endif  // V2MESH_STEP_H
