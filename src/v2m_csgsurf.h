// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_csgsurf.h -- shape constructs (JSON: MCX Shapes, JMesh Shape* / CSG*) ->
// their exact boundary surface, for --mode cdt: every crease, corner and knife
// edge a chain of edges of the surface, so the tets keep them exactly.
//
//   1. each primitive's surfaces as rectangular patches of their parameter
//      planes (a box: 6 planes; a sphere: 2 halves, their seams and poles
//      shared edges; a cylinder / cone: 2 half-sides and 2 half-disk caps; a
//      torus: 4 quarters), each edge sampled once, shared by its patches;
//   2. where two primitives' surfaces cross: marched in the parameter plane of
//      one's patches (the other's signed distance), each point projected onto
//      both surfaces; the same nodes taken into the other's patches (split
//      where it crosses their edges), and split where two such curves meet
//      (three surfaces: a triple point, on all three);
//   3. each patch triangulated in its plane with its edges and curves as
//      constraints (v2m_plc.h cdt2d) and interior points (--size);
//   4. each piece between the curves kept if the composed labels (the scene's
//      own fields: its CSG, clipping and --overlap rule) differ across it, the
//      two labels its inner / outer ones.
//
// Supported: boxes (Grid, Box, Subgrid, ShapeBox3), spheres, cylinders, cones
// and frusta, tori. Planes, slabs, layers, ellipsoids and lenses are not yet;
// neither are two surfaces lying on each other (a box's face on another's) --
// those throw, and --mode mesh (the particle mesher) meshes them.

#ifndef V2MESH_CSGSURF_H
#define V2MESH_CSGSURF_H

#include <cstddef>
#include <string>
#include <vector>

#include "v2m_mesh.h"
#include "v2m_sdfshape.h"

namespace tn {

struct CsgSurfOptions {
    double tol = 0;      // chord tolerance (mm); 0 = 0.05 % of the domain's size
    double angle = 15;   // max turn of a segment (degrees)
    double size = 0;     // max edge length (mm; --size); 0 = the curvature alone
};

struct CsgSurfStats {
    size_t primitives = 0, patches = 0, curves = 0, triple = 0;
    size_t pieces = 0, kept = 0, triangles = 0, nodes = 0;
    double tol = 0;
};

// The exact boundary surface of scene `sc` (load_shapes): triangles with inner /
// outer labels, world coordinates. Throws on an unsupported construct.
Mesh csg_surface(const ShapeScene& sc, const CsgSurfOptions& o = CsgSurfOptions(), CsgSurfStats* st = nullptr);

}  // namespace tn

#endif  // V2MESH_CSGSURF_H
