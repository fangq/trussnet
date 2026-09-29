// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_shapes.h -- synthetic multi-label test volumes (label 0 = exterior): the
// benchmark shapes of Walton's thesis (corrugated sphere, gyroid, triple helix,
// box + hemispheres, hemisphere sandwich) plus concave and thin cases (torus,
// U-bend, hollow ball, thin slab), junction cases (T-junction, two balls) and
// nested shells. Unit voxels on an n^3 grid spanning (-1,1)^3.

#ifndef V2MESH_SHAPES_H
#define V2MESH_SHAPES_H

#include <string>
#include <vector>

#include "v2m_volume.h"

namespace tn {

std::vector<std::string> shape_names();
LabelVolume make_shape(const std::string& name, int n);   // throws on an unknown name

}  // namespace tn

#endif  // V2MESH_SHAPES_H
