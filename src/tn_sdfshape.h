// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_sdfshape.h -- meshes from shape constructs (JSON): MCX's "Shapes" list
// (Grid, Box, Subgrid, Sphere, Cylinder, X/Y/ZSlabs, X/Y/ZLayers, Lens) and
// JMesh's shape primitives (ShapeBox3, ShapeSphere, ShapeCylinder,
// ShapeEllipsoid, ShapeTorus, ShapeCone, ShapeConeFrustum, ShapeSphereShell,
// ShapeSphereSegment, ShapePlane3) with CSG (CSGObject, CSGUnion,
// CSGIntersect, CSGSubtract; operands inline or by "(name)").
//
// Each shape is a signed distance function s (> 0 inside; union = max,
// intersection = min, subtraction = min(a, -b)). The objects form one sequence,
// sequentially overwriting: object 1 is the outermost shape -- everything
// outside it is exterior (label 0) -- and each later object overwrites what is
// before it, cut to object 1 ("Clip": false: not cut; the domain is then all
// the objects' bounds). Object i's region: s'_i = min(s_i, s_1, -max_{j>i} s_j);
// label l's field s_l = max over its objects; the exterior's -max_l s_l.
//
// The labels' fields are compiled into one program (tn_sdf_body.cl) that the
// mesher evaluates exactly, on the host and the device.

#ifndef TRUSSNET_SDFSHAPE_H
#define TRUSSNET_SDFSHAPE_H

#include <array>
#include <string>
#include <vector>

#include "tn_tpm.h"

namespace tn {

struct ShapeScene {
    std::vector<float> prog;          // the compiled label fields (tn_sdf_body.cl layout)
    int nlab = 0;                     // labels 0 .. nlab - 1
    std::array<double, 3> lo{ { 0, 0, 0 } }, hi{ { 0, 0, 0 } };   // the domain (world)
    std::vector<std::string> objects; // one line per object (for the log)
    // sharp features of the primitives (world): 1 corner x y z; 2 segment p0 p1;
    // 3 circle c n r -- candidates for pinned nodes (tn_particles.h), kept where
    // the composed labels differ round them
    std::vector<float> feat;
    bool clip = true;
};

// Does the file / text hold shape constructs (a "Shapes" key, or Shape* / CSG*
// keys at the top)?
bool is_shape_json_file(const std::string& path);

// Parse shape constructs (a file path, or JSON text starting with '{' or '[').
// Throws std::runtime_error on errors.
ShapeScene load_shapes(const std::string& path_or_text, bool clip_default = true);

// label l's field at point p of the program's frame (the mesher's grid mm)
float sdf_eval(const std::vector<float>& prog, int l, const float* p);

// label l's field at world point p (s > 0 inside)
double scene_sdf(const ShapeScene& sc, int l, const double* p);

// The fields on a raster of spacing `voxel` over the domain (+ a margin): p_l =
// clamp(0.5 + s_l / w), w = 1.5 voxel (as tn_remesh.h). The program's origin is
// set to the raster's, so the mesher's grid-mm coordinates index it directly.
Tpm rasterize_scene(ShapeScene& sc, double voxel);

// The largest principal curvature of the primitives' own surfaces passing within
// `band` of each voxel of an nx x ny x nz raster of spacing `voxel` (grid mm, the
// program's frame): 1/R of spheres, cylinder sides, tori (their tubes), cones
// (locally); 0 of boxes, planes, slabs, caps -- for the sizing
std::vector<float> sdf_curvature(const std::vector<float>& prog, int nx, int ny, int nz, double voxel, double band);

}  // namespace tn

#endif  // TRUSSNET_SDFSHAPE_H
