// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_sdfshape.h -- meshes from shape constructs (JSON): MCX's "Shapes" list
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
// Other overlap rules (--overlap; object 1, the container when cut to it,
// always yields to the others): nest (the smaller object wins an overlap),
// max / min (the higher / lower label), order:L1,L2,.. (the first listed),
// union (overlapping objects are one region, of the first one's label), cells
// (each overlap a region of its own, a new label after the largest Tag), and
// split (the halfway surface s_a = s_b between two that overlap).
//
// The labels' fields are compiled into one program (v2m_sdf_body.cl) that the
// mesher evaluates exactly, on the host and the device.

#ifndef V2MESH_SDFSHAPE_H
#define V2MESH_SDFSHAPE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "v2m_tpm.h"

namespace tn {

// a region: inside all of `in` (in[0] its own object), outside all of `out`,
// and, split, nearer the inside of in[0] than of each of `split`
// (s_0 - s_j >= 0); of label `tag`
struct ShapeRegion {
    std::vector<int> in, out, split;
    int tag = 0;
};

struct ShapeScene {
    std::vector<float> prog;          // the compiled label fields (v2m_sdf_body.cl layout)
    int nlab = 0;                     // labels 0 .. nlab - 1
    std::array<double, 3> lo{ { 0, 0, 0 } }, hi{ { 0, 0, 0 } };   // the domain (world)
    std::vector<std::string> objects; // one line per object (for the log)
    // sharp features of the primitives (world): 1 corner x y z; 2 segment p0 p1;
    // 3 circle c n r; 4 polyline n p0 .. p(n-1) (two objects' surfaces
    // crossing) -- candidates for pinned nodes (v2m_particles.h), kept where the
    // composed labels differ round them
    std::vector<float> feat;
    // the objects, compiled (their CSG trees' code), with their bounds (world;
    // infinite where unbounded) and labels: for the per-brick programs
    std::vector<std::vector<float>> ocode;
    std::vector<std::array<double, 6>> obox;
    std::vector<int> otag;
    // the primitives (every object's CSG leaves), compiled, with their bounds:
    // for the creases where two of their surfaces cross (between objects, or
    // inside one object's CSG)
    std::vector<std::vector<float>> pcode;
    std::vector<std::array<double, 6>> pbox;
    std::vector<ShapeRegion> regions;  // the labels' regions (by the overlap rule)
    std::string overlap = "overwrite";
    bool clip = true;
    size_t brick_programs = 0;        // (build_brick_programs: distinct programs made)
};

// Does the file / text hold shape constructs (a "Shapes" key, or Shape* / CSG*
// keys at the top)?
bool is_shape_json_file(const std::string& path);

// Parse shape constructs (a file path, or JSON text starting with '{' or '[').
// Throws std::runtime_error on errors.
// overlap: how overlapping objects share (overwrite nest split max min union
// cells order:L1,L2,..).
ShapeScene load_shapes(const std::string& path_or_text, bool clip_default = true,
                       const std::string& overlap = "overwrite");

// label l's field at point p of the program's frame (the mesher's grid mm)
float sdf_eval(const std::vector<float>& prog, int l, const float* p);

// label l's field at world point p (s > 0 inside)
double scene_sdf(const ShapeScene& sc, int l, const double* p);

// The fields on a raster of spacing `voxel` over the domain (+ a margin): p_l =
// clamp(0.5 + s_l / w), w = 1.5 voxel (as v2m_remesh.h). The program's origin is
// set to the raster's, so the mesher's grid-mm coordinates index it directly.
Tpm rasterize_scene(ShapeScene& sc, double voxel);

// Per-brick programs (v2m_sdf_body.cl): for each 8^3-voxel brick of the raster
// rasterize_scene makes at `voxel`, the labels' code with only the objects whose
// bounds come within the cull margin of it (the rest as the culled constants);
// bricks alike share one. The evaluator runs its point's brick's. Set the cull
// margin (prog[7 + N]) first. Returns the programs made.
size_t build_brick_programs(ShapeScene& sc, double voxel);

// The largest principal curvature of the primitives' own surfaces passing within
// `band` of each voxel of an nx x ny x nz raster of spacing `voxel` (grid mm, the
// program's frame): 1/R of spheres, cylinder sides, tori (their tubes), cones
// (locally); 0 of boxes, planes, slabs, caps -- for the sizing
std::vector<float> sdf_curvature(const std::vector<float>& prog, int nx, int ny, int nz, double voxel, double band);

// The points where a feature curve (feat: grid mm, ShapeScene::feat layout)
// meets another interface -- the labels round it (6 samples at eps) change along
// the curve -- and the feature corners with 3+ labels round them: where the
// geometry crowds (an edge piercing a surface), for a finer sizing there
std::vector<float> sdf_feature_points(const std::vector<float>& prog, const std::vector<float>& feat, float eps);

// The points of the feature curves (feat: grid mm) where the regions meet at an
// acute angle -- a knife edge (a box less a larger sphere: its holes' rims) or a
// thin notch: round each (every `step` along the curve) the labels on a circle of
// radius eps across it, and the narrowest sector one label holds. (x, y, z, its
// angle in degrees) each, for the angles below max_deg -- for a finer sizing
// there, as a wedge thinner than an element is lost
std::vector<float> sdf_acute_points(const std::vector<float>& prog, const std::vector<float>& feat, float eps, float step,
                                    float max_deg);

// The local thickness (voxels) of each voxel's own region (label L[v]): twice
// the largest value its field reaches within R voxels -- a layer thinner than
// 2 R voxels shows its thickness, a thicker region >= 2 R -- for the thin-layer
// sizing
std::vector<float> sdf_thickness(const std::vector<float>& prog, const uint16_t* L, int nx, int ny, int nz, double voxel, int R);

}  // namespace tn

#endif  // V2MESH_SDFSHAPE_H
