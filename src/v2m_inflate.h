/***************************************************************************//**
**  \file    v2m_inflate.h
**  \brief   Inflated cortical / white-matter surfaces, with the mesh around them moved along
**
**  As FreeSurfer's mris_inflate: the boundary of a set of labels (e.g. one cerebral
**  hemisphere -> its pial surface; the white matter -> the white surface) is smoothed --
**  implicit steps, (I + tau L) X = X_old, each over ~sqrt(tau) edges -- with the edges
**  relaxed back toward their first lengths in the surface's tangent plane after each step (the
**  local area redistributed: the sulcal walls keep their share), and every closed surface
**  rescaled to its first area (FreeSurfer's: the inflated surface grows past the brain) or its
**  first volume (it keeps its size: the fixed head around a moving mesh). The gyri and sulci
**  flatten out; the vertices keep their identity. A vertex where the surface pinches (two
**  sheets meeting at an edge or a vertex) is split into one per sheet first, so each sheet
**  smooths on its own. The sulcal depth ("sulc", positive in the sulci) is each vertex's
**  displacement along the outward normal of the surface once smoothed by sulc_tau (the folds
**  gone, the lobes kept).
**
**  A tetrahedral mesh may move with its surfaces (a moving mesh): after each step the free
**  nodes (neither on an inflated surface nor on the mesh's outer boundary, which stays put)
**  take the minimizer of sum w_ij |x_i - x_j|^2, w_ij = 1 / (the edge's current length) --
**  a harmonic extension re-weighted on the deformed geometry -- then the nodes of any
**  inverted or nearly flat tet are pushed out along its volume gradient (untangling); where
**  tets stay inverted the step is taken back locally. Connectivity and labels are kept: the
**  regions no longer match the input volume.
*******************************************************************************/

#ifndef V2MESH_INFLATE_H
#define V2MESH_INFLATE_H

#include <string>
#include <vector>

#include "v2m_mesh.h"

namespace tn {

// one inflated surface: "name=L1+L2..[@L|@R][-E1-E2..]"
struct InflateSet {
    std::vector<int> labels;    // the region (its boundary is the surface)
    std::string name;           // the output's JGIFTI surface GIFTIData.<name>
    int side = 0;               // 0 all; -1 / +1 the half left / right of the region's midline plane
    std::vector<int> exclude;   // labels whose part of the region leaves it: the region's tets
                                // reached first from them, not from the seed, growing through
                                // the tets (e.g. -5 drops the cerebellar WM and the lower
                                // brainstem from a cerebrum given as 1+2+..)
    int seed = -1;              // "^L": the exclusion's other source (default: the first label; it
                                // may lie outside the region -- a white surface 2+..-5^1 grows
                                // from the cerebral cortex, its WM's own label seeding every WM)
};

struct InflateParams {
    std::vector<InflateSet> sets;
    int iters = 6;                 // smoothing steps (x tau 50: the lobes as clear as FreeSurfer's)
    double tau = 50.0;             // > 0: implicit steps (each smooths over ~sqrt(tau) edges); 0: explicit
    bool cot = false;              // the implicit steps' weights: cotangent (mean curvature flow; the
                                   // triangles keep their squeezed spread), else uniform (they even out)
    int cot_finish = 2;            // surfaces alone: cotangent steps at the end (the needles out)
    double cot_finish_tau = 5.0;   // their tau
    double sulc_tau = 150.0;       // sulc measured against the surface smoothed this far (sum of tau)
    int sulc_hp = 0;               // sulc high-passed: minus its mean over ~this many rings (0: off; the
                                   // FreeSurfer-style sum needs none -- the depth-below-a-surface did)
    int curv_avg = 10;             // curv averaged over this many rings (the voxel-scale texture out)
    int sulc_fs = 1;               // 1: sulc as FreeSurfer's -- each step's move along the normal, summed
                                   // over the inflation; 0: the depth below the sulc_tau surface
    int presmooth = 10;            // Taubin passes on the surface first (FreeSurfer's smoothwm: the
                                   // voxel staircase out); sulc, curv and the edges' lengths from it
    double dist = 0.5;             // weight of the tangential edge relaxation (0: none)
    int relax = 10;                // its passes after each step
    int keep = 0;                  // each closed surface keeps: 1 its area, 2 its volume, 0 automatic
                                   // (area for surfaces alone, volume for a moving mesh)
    int finish = 10;               // surfaces alone: Taubin passes at the end (the small cones out)
    int substep = 1;               // steps between the moves of the tetrahedral mesh
    int averages = 0;              // explicit steps: the displacement averaged this many times
    bool move_mesh = false;        // true: the tets move with the surfaces (a moving mesh; experimental:
                                   // big steps invert tets), else only the surfaces inflate
    bool verbose = false;
};

struct InflateStats {
    size_t surf_nodes = 0, surf_faces = 0, components = 0, split = 0;   // split: pinch copies added
    size_t fixed_nodes = 0, free_nodes = 0;   // (the tetrahedral mesh: outer boundary, moved)
    size_t excluded = 0;                      // tets an exclusion took out of the regions
    size_t cleaned = 0;                       // tets the regions' cleaning flipped (spikes, notches)
    size_t folds = 0;                         // faces still folded after the last step's unfolding
    int iters_done = 0, shortened = 0;        // steps made; domain steps bisected
    bool stopped = false;                     // a step could not be made without inverting a tet
    bool by_area = true;                      // what the closed surfaces kept
    double area = 0.0;                        // the surfaces' total area at the start, mm^2
    double volume = 0.0;                      // the volume the closed ones enclose, mm^3
    size_t closed = 0;                        // the closed surfaces
    double sulc_min = 0.0, sulc_max = 0.0;
    double min_vol_ratio = 1.0;               // the smallest tet volume / its first volume
    double ms = 0.0;
};

// "lh=1+2+4@L-5,white=2" -> two sets ('+' joins labels, ',' separates the sets, "name="
// names one -- unnamed: surface1, .. --, "@L" / "@R" keeps a half, "-E" an exclusion);
// the SIAM presets "white" and "pial" (or "gm") each give lh_* and rh_*
std::vector<InflateSet> parse_inflate_sets(const std::string& s);

// Inflate m in place: with move_mesh, m.nodes deformed; m.surfaces each set's inflated surface
// (its nodes, triangles, sulc and curv). m holds tets (+ labels) or labelled triangles (MeshTri
// inner / outer). Throws if no set has a boundary in it.
void inflate_mesh(Mesh& m, const InflateParams& p, InflateStats& st);

}  // namespace tn

#endif  // V2MESH_INFLATE_H
