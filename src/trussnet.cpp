// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// trussnet.cpp -- command-line driver: multi-label volume -> sizing field ->
// graded hex particles -> truss relaxation -> tessellation (stages added
// incrementally; see the plan in README.md).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tn_jmesh.h"
#include "tn_mesh.h"
#include "tn_meshio.h"
#include "tn_modes.h"
#include "tn_remesh.h"
#include "tn_cdt.h"
#include "tn_log.h"
#include "tn_2d.h"
#include "tn_pipeline.h"
#include "tn_tpm.h"
#include "tn_shapes.h"
#include "tn_volume.h"

namespace {

struct Config {
    std::string input, shape, output;
    std::string mode = "mesh";        // --mode: mesh (default) surface points check
    bool faces = false;               // --faces: also write the region surfaces (MeshTri)
    int opt_rounds = 3;               // --opt-rounds (--mode optimize)
    std::string image;                // --image: the volume beside a point / mesh input
    double raster_voxel = 0;          // --raster-voxel (--mode remesh / repair); 0 = automatic
    bool exact_tess = false;          // --exact-tess: surface / repair with the full tessellation
    double cdt_fill = -1;             // --cdt-fill (--mode cdt): interior point spacing; 0 none, < 0 automatic
    int dim = 96;
    tn::PipelineOptions o;
    tn::TpmOptions tpm;
    std::vector<std::string> given;   // the flags set on the command line (2-D mode keeps its own defaults)
};

std::vector<int> int_list(const std::string& v) {
    std::vector<int> r;
    size_t p0 = 0;

    while (p0 <= v.size()) {
        const size_t p1 = v.find(',', p0);
        r.push_back(std::atoi(v.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0).c_str()));

        if (p1 == std::string::npos) {
            break;
        }

        p0 = p1 + 1;
    }

    return r;
}

void usage(const char* exe) {
    std::fprintf(stderr,
                 "trussnet " TN_VERSION " -- GPU particle (truss) multi-label tetrahedral mesher\n"
                 "usage: %s (-i volume.{nii,nii.gz,jnii,bnii} | --shape NAME [--dim N]) [options]\n"
                 "  -o FILE          output mesh (.jmsh text / .bmsh binary)\n"
                 "  --mode M         what to run and write (default mesh):\n"
                 "                     mesh     a volume -> labelled tets (--faces: + the region surfaces)\n"
                 "                     surface  a volume -> only its region / exterior surfaces (MeshTri: v1 v2 v3\n"
                 "                              inner outer; outer 0 = the exterior)\n"
                 "                     points   a volume -> the relaxed nodes, before tessellation (MeshNode +\n"
                 "                              NodeLabel / NodeType / NodePartner)\n"
                 "                     check    -i a mesh or surface (.jmsh .bmsh .off .stl) -> a report: quality,\n"
                 "                              open / junction edges, self-intersections (exit 3 on problems)\n"
                 "                     optimize -i a labelled tet mesh (.jmsh .bmsh) -> the same regions, better\n"
                 "                              tets (flips, kites, collapses, Steiner points, smoothing; the\n"
                 "                              interfaces and the boundary are kept; -q guards the radius-edge)\n"
                 "                     tessellate -i points (.xyz [x y z label], .off, .jmsh) -> their Delaunay tets\n"
                 "                              (the convex hull; labels: the nodes' most frequent); with --image\n"
                 "                              and trussnet's labelled nodes (--mode points), the mesher's full\n"
                 "                              tessellation (labels, conformity repair, -q, ODT, optimiser)\n"
                 "                     cdt      -i closed, non-self-intersecting surfaces (.jmsh .bmsh .off .stl) -> labelled\n"
                 "                              tets with the surfaces kept exactly (constrained Delaunay; regions as\n"
                 "                              remesh; then the optimiser unless --opt 0)\n"
                 "                     remesh   -i closed surfaces (.jmsh .off .stl; may self-intersect, overlap or be\n"
                 "                              oriented either way) -> labelled tets of the regions they enclose:\n"
                 "                              rasterized into per-region soft fields, then the whole mesher\n"
                 "                              (regions: MeshTri inner / outer labels, else nested shells)\n"
                 "                     repair   as remesh, writing the region surfaces: clean, closed, no\n"
                 "                              self-intersections\n"
                 "  --faces          mesh / tessellate: also write the region surfaces (MeshTri) with the tets\n"
                 "  --exact-tess     surface / repair: tessellate every node and run the quality stages (the\n"
                 "                   default tessellates only the surface nodes: about twice as fast)\n"
                 "  --cdt-fill H     --mode cdt: interior points on a lattice of spacing H inside the regions\n"
                 "                   (default --size, else 1.5 x the surface's mean edge; 0 = none)\n"
                 "  --raster-voxel V remesh / repair: the raster spacing (default: the smaller of --size / 3\n"
                 "                   (else extent / 160) and half the input's mean edge)\n"
                 "  --image FILE     --mode tessellate: the volume the nodes came from (or --shape NAME)\n"
                 "  --opt-rounds N   --mode optimize: rounds of the optimiser (default 3)\n"
                 "  --size MM        default element size (default 3 x voxel)\n"
                 "  --hmin MM        smallest element size (default size/3)\n"
                 "  --hmax MM        largest element size (default size)\n"
                 "  --K K            elements per radian of curvature (default 3)\n"
                 "  --grad G         sizing gradient limit (default 0.3)\n"
                 "  --sigma S        indicator smoothing (voxels, default 1)\n"
                 "  --sigma-thin S   interface smoothing in thin layers (< 2-4 voxels; default 0.35, 0 = off)\n"
                 "  --thick B        thin layers: h <= local thickness / B (0 = off)\n"
                 "  --thin-floor V   smallest thin-layer size, voxels (default 0.5)\n"
                 "  --preserve M     keep each voxel's own label on top of the smoothed fields\n"
                 "                   by margin M (0 = off)\n"
                 "  --nseed N        coarse seeding levels (default 8)\n"
                 "  --iters N        max relaxation iterations (default 500)\n"
                 "  --fscale F       rest length / h (default 1.2)\n"
                 "  --fsurf F        rest length / h between interface nodes (default 1.0)\n"
                 "  --dt T           Jacobi relaxation factor, and FIRE's first step (default 0.5)\n"
                 "  --snap S         interior nodes within S*h of an interface join it (default 0.5)\n"
                 "  --relax M        relaxation step: fire (default; inertial, adaptive time step, FIRE,\n"
                 "                   Bitzek et al. 2006) or jacobi; --trap voxel always uses jacobi\n"
                 "  --fire-dtmax X   FIRE: largest time step, X times the first (default 2)\n"
                 "  --dptol T        stop relaxing when 99%% of the nodes move < T h per step (default\n"
                 "                   0.002; 0 = run all --iters)\n"
                 "  --thin B         seed thinning: before relaxing, drop each seed that has a kept one\n"
                 "                   of its interface / label closer than B*h (e.g. 0.7: coarser thin\n"
                 "                   layers next to fine interfaces, fewer nodes; default 0 = off)\n"
                 "  --lsize L:H,..   per-label element size (mm), e.g. 1:4,2:3 (default --size)\n"
                 "  --isize H|L:H|A:B:H,..  element size (mm) at interfaces only: every interface,\n"
                 "                   every interface of label L (0 = the outer surface), or the A|B\n"
                 "                   interface; --size / --lsize set the interiors, --grad the grading\n"
                 "                   (e.g. --size 6 --isize 0:2,3:4:1.5)\n"
                 "  --thresholds T1,T2,..  gray-scale input: label = number of thresholds <= intensity;\n"
                 "                   the interfaces are the iso-surfaces (0 = below T1 = exterior)\n"
                 "  --gray-sigma S   Gaussian pre-smoothing of the gray-scale input (voxels)\n"
                 "  a 4-D input (.jnii/.bnii/.nii[.gz]) is a tissue-probability map (TPM): the labels\n"
                 "  are the argmax of the class probabilities (exterior pockets filled)\n"
                 "  --tpm-exterior C,..  exterior channels (0-based; default: the ones named\n"
                 "                   background/air/bg/outside, else exterior = 1 - sum(tissues))\n"
                 "  --tpm-map L0,L1,..  label of each channel (0 = exterior; shared = summed)\n"
                 "  --tpm-spm6       merge the 18 siamize classes to SPM6 (GM WM CSF Bone Soft)\n"
                 "  --tpm-sigma S    Gaussian smoothing of the probabilities (voxels, default 0)\n"
                 "  --tpm-thresh T|L:T,..  per-label threshold (default 0.5 = the argmax): label =\n"
                 "                   argmax(p_l - t_l + 0.5), the fields (--tpm-fields) shifted alike, so\n"
                 "                   the a|b interface is at p_a - t_a = p_b - t_b; lower t_l grows label\n"
                 "                   l. T alone: every tissue label (vs the exterior, label 0)\n"
                 "  --tpm-fields     interfaces from the probabilities (smoothed p_a = p_b) instead\n"
                 "                   of the argmax labels' smoothed indicators\n"
                 "  --tpm-holes      keep the enclosed exterior pockets (default: filled with the\n"
                 "                   nearest tissue, as brain2mesh)\n"
                 "  --gpu [N]        relax on OpenCL device N (default: the first GPU)\n"
                 "  --jseed C        junction-line seeds, one per C x spacing cell (default 0.8, 0 = off)\n"
                 "  --no-corners     no fixed nodes where >= 4 labels meet\n"
                 "  --trap M         boundary trapping: smooth (sub-voxel interface, default) or\n"
                 "                   voxel (exact voxel faces: DDA walk + nearest staircase face)\n"
                 "  -q Q             max radius-edge ratio (TetGen / gpu_brain2mesh -q; default 2.0, 0 = off):\n"
                 "                   worse tets get their circumcentre inserted (on the interface if it\n"
                 "                   encroaches), and the optimiser may not create one\n"
                 "  --opt 0|1        sliver repair: 3-2/2-3 flips, collapses, Steiner points (default 1)\n"
                 "  --smooth N       quality-guarded ODT passes over the interior nodes (default 5)\n"
                 "  --repair N       max restricted-Delaunay repair rounds (default 6)\n"
                 "  --dump-grid F    write labels / h / grade to a BJData .bnii (debug)\n"
                 "  --dump-nodes F   write the relaxed nodes (+label, type) to BJData (debug)\n"
                 "  -v               progress\n"
                 "  --version        print the version\n"
                 "shapes:", exe);

    for (const std::string& s : tn::shape_names()) {
        std::fprintf(stderr, " %s", s.c_str());
    }

    std::fprintf(stderr, "\n");
}

// the command line into cfg: -1 to go on, else the exit code (--help, --version,
// a bad value)
int parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        cfg.given.push_back(a);
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "trussnet: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "--version") {
            std::printf("trussnet %s\n", TN_VERSION);
            return 0;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (a == "-i") {
            cfg.input = next();
        } else if (a == "--shape") {
            cfg.shape = next();
        } else if (a == "--dim") {
            cfg.dim = std::atoi(next());
        } else if (a == "-o") {
            cfg.output = next();
        } else if (a == "--size") {
            cfg.o.grid.hbase = static_cast<float>(std::atof(next()));
        } else if (a == "--hmin") {
            cfg.o.grid.hmin = static_cast<float>(std::atof(next()));
        } else if (a == "--hmax") {
            cfg.o.grid.hmax = static_cast<float>(std::atof(next()));
        } else if (a == "--K") {
            cfg.o.grid.K = static_cast<float>(std::atof(next()));
        } else if (a == "--grad") {
            cfg.o.grid.g = static_cast<float>(std::atof(next()));
        } else if (a == "--sigma") {
            cfg.o.grid.sigma = static_cast<float>(std::atof(next()));
        } else if (a == "--dump-grid") {
            cfg.o.dump_grid = next();
        } else if (a == "--dump-nodes") {
            cfg.o.dump_nodes = next();
        } else if (a == "--nseed") {
            cfg.o.relax.nseed = std::atoi(next());
        } else if (a == "--iters") {
            cfg.o.relax.max_iters = std::atoi(next());
        } else if (a == "--fscale") {
            cfg.o.relax.fscale = static_cast<float>(std::atof(next()));
        } else if (a == "--dt") {
            cfg.o.relax.dt = static_cast<float>(std::atof(next()));
        } else if (a == "--fsurf") {
            cfg.o.relax.fsurf = static_cast<float>(std::atof(next()));
        } else if (a == "--sigma-thin") {
            cfg.o.grid.sigma_thin = static_cast<float>(std::atof(next()));
        } else if (a == "--thick") {
            cfg.o.grid.thick = static_cast<float>(std::atof(next()));
        } else if (a == "--thin-floor") {
            cfg.o.grid.thin_floor = static_cast<float>(std::atof(next()));
        } else if (a == "--preserve") {
            cfg.o.grid.preserve = static_cast<float>(std::atof(next()));
        } else if (a == "--thin") {   // node thinning: drop nodes closer than B h to a kept one
            cfg.o.relax.thin = static_cast<float>(std::atof(next()));
        } else if (a == "--isize") {   // interface sizes: H | L:H | A:B:H [,...] (mm)
            cfg.o.grid.isize.parse(next());
        } else if (a == "--lsize") {   // per-label element size: L:H[,L:H...] (mm)
            std::string v = next();
            size_t p0 = 0;

            while (p0 < v.size()) {
                size_t p1 = v.find(',', p0);
                const std::string item = v.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
                const size_t c = item.find(':');

                if (c == std::string::npos) {
                    throw std::runtime_error("--lsize wants L:H[,L:H...]");
                }

                const int l = std::atoi(item.substr(0, c).c_str());

                if (l < 0 || l > 65535) {
                    throw std::runtime_error("--lsize: bad label");
                }

                if (static_cast<int>(cfg.o.grid.hlab.size()) <= l) {
                    cfg.o.grid.hlab.resize(l + 1, 0.0f);
                }

                cfg.o.grid.hlab[l] = static_cast<float>(std::atof(item.substr(c + 1).c_str()));

                if (p1 == std::string::npos) {
                    break;
                }

                p0 = p1 + 1;
            }
        } else if (a == "--thresholds") {   // gray-scale iso-values: t1,t2,...
            cfg.o.thresholds.clear();
            std::string v = next();
            size_t p0 = 0;

            while (p0 <= v.size()) {
                const size_t p1 = v.find(',', p0);
                cfg.o.thresholds.push_back(static_cast<float>(std::atof(v.substr(p0, p1 - p0).c_str())));

                if (p1 == std::string::npos) {
                    break;
                }

                p0 = p1 + 1;
            }
        } else if (a == "--tpm-exterior") {
            cfg.tpm.exterior = int_list(next());
        } else if (a == "--tpm-map") {
            cfg.tpm.map = int_list(next());
        } else if (a == "--tpm-spm6") {
            cfg.tpm.spm6 = true;
        } else if (a == "--tpm-fields") {
            cfg.tpm.fields = true;
        } else if (a == "--tpm-holes") {
            cfg.tpm.fill_holes = false;
        } else if (a == "--tpm-sigma") {
            cfg.tpm.sigma = static_cast<float>(std::atof(next()));
        } else if (a == "--tpm-thresh") {   // T | L:T [,...]: per-label threshold bias
            tn::parse_tpm_thresh(next(), cfg.tpm);
        } else if (a == "--gray-sigma") {
            cfg.o.gray_sigma = static_cast<float>(std::atof(next()));
        } else if (a == "--gpu") {   // optional device index
            cfg.o.gpu = -1;

            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.o.gpu = std::atoi(argv[++i]);
            }
        } else if (a == "--jseed") {
            cfg.o.relax.jseed = static_cast<float>(std::atof(next()));
        } else if (a == "--no-corners") {
            cfg.o.relax.corners = false;
        } else if (a == "--trap") {
            const std::string m = next();

            if (m != "smooth" && m != "voxel") {
                throw std::runtime_error("--trap wants smooth or voxel");
            }

            cfg.o.relax.voxel_trap = m == "voxel";
        } else if (a == "--mode") {
            const std::string m = next();

            if (m != "mesh" && m != "surface" && m != "points" && m != "check" && m != "optimize" && m != "tessellate" &&
                    m != "remesh" && m != "repair" && m != "cdt") {
                throw std::runtime_error("--mode wants mesh, surface, points, tessellate, optimize, cdt, remesh, repair or check");
            }

            cfg.mode = m;
        } else if (a == "--faces") {
            cfg.faces = true;
        } else if (a == "--exact-tess") {
            cfg.exact_tess = true;
        } else if (a == "--cdt-fill") {
            cfg.cdt_fill = std::atof(next());
        } else if (a == "--raster-voxel") {
            cfg.raster_voxel = std::atof(next());
        } else if (a == "--image") {
            cfg.image = next();
        } else if (a == "--opt-rounds") {
            cfg.opt_rounds = std::atoi(next());
        } else if (a == "--relax") {
            const std::string m = next();

            if (m != "jacobi" && m != "fire") {
                throw std::runtime_error("--relax wants jacobi or fire");
            }

            cfg.o.relax.fire = m == "fire";
        } else if (a == "--dptol") {
            cfg.o.relax.dptol = static_cast<float>(std::atof(next()));
        } else if (a == "--fire-dtmax") {
            cfg.o.relax.fire_dtmax = static_cast<float>(std::atof(next()));
        } else if (a == "--snap") {
            cfg.o.relax.snap = static_cast<float>(std::atof(next()));
        } else if (a == "-q" || a == "--quality" || a == "--reratio") {
            cfg.o.q = std::atof(next());
        } else if (a == "--opt") {
            cfg.o.opt = std::atoi(next()) != 0;
        } else if (a == "--smooth") {
            cfg.o.smooth = std::atoi(next());
        } else if (a == "--repair") {
            cfg.o.max_repair = std::atoi(next());
        } else if (a == "-v") {
            cfg.o.relax.verbose = true;
        } else {
            std::fprintf(stderr, "trussnet: unknown argument '%s'\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    int rc = -1;

    try {
        rc = parse_args(argc, argv, cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "trussnet: %s\n", e.what());
        rc = 2;
    }

    if (rc >= 0) {
        return rc;
    }

    if (cfg.input.empty() && cfg.shape.empty() && cfg.image.empty()) {
        usage(argv[0]);
        return 2;
    }

    typedef std::chrono::steady_clock clk;
    auto ms = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
    };

    if (cfg.mode == "check") {   // a mesh or a surface: a report, no meshing
        if (cfg.input.empty()) {
            std::fprintf(stderr, "trussnet: --mode check wants -i MESH (.jmsh .bmsh .off .stl)\n");
            return 2;
        }

        try {
            const tn::Mesh m = tn::read_mesh(cfg.input);
            const tn::MeshReport rep = tn::check_mesh(m);
            tn::print_report(rep, cfg.input);
            return rep.ok() ? 0 : 3;
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
            return 1;
        }
    }

    if (cfg.mode == "optimize") {   // a labelled tet mesh: the optimiser alone
        if (cfg.input.empty()) {
            std::fprintf(stderr, "trussnet: --mode optimize wants -i MESH (.jmsh .bmsh with MeshElem)\n");
            return 2;
        }

        try {
            tn::Mesh m = tn::read_mesh(cfg.input);
            tn::print_report(tn::check_mesh(m), cfg.input + " (before)");
            tn::OptParams op;
            op.q = cfg.o.q;
            op.max_rounds = cfg.opt_rounds;
            op.verbose = cfg.o.relax.verbose;
            tn::OptStats os;
            const auto t0 = clk::now();
            tn::optimize_tets(m, op, os);
            TN_FPRINTF(stderr, "[opt]   %d 3-2 + %d 2-3 flips, %d kites flattened, %d collapses, %d Steiner points, %d moves "
                       "(%d rounds, %.0f ms)\n", os.flips32, os.flips23, os.kites, os.collapses, os.steiner, os.moves,
                       os.rounds, ms(t0));
            const tn::MeshReport after = tn::check_mesh(m);
            tn::print_report(after, "optimised");

            if (!cfg.output.empty()) {
                tn::write_jmesh_auto(cfg.output, m);
                TN_FPRINTF(stderr, "[output] %s written: %zu nodes, %zu tets\n", cfg.output.c_str(), m.nodes.size() / 3,
                           m.tets.size() / 4);
            }

            return after.ok() ? 0 : 3;
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
            return 1;
        }
    }

    if (cfg.mode == "cdt") {   // surfaces -> constrained Delaunay tets
        if (cfg.input.empty()) {
            std::fprintf(stderr, "trussnet: --mode cdt wants -i SURFACES (.jmsh .bmsh .off .stl)\n");
            return 2;
        }

        try {
            const tn::Mesh surf = tn::read_mesh(cfg.input);
            tn::Mesh m;
            tn::CdtStats cs;
            tn::OptStats os;
            const double fill = tn::run_cdt(surf, cfg.o, cfg.cdt_fill, cfg.opt_rounds, m, cs, os);
            TN_FPRINTF(stderr, "[cdt]   %zu vertices, %zu triangles (%zu junction edges) + %zu interior points (spacing %.4g) -> "
                       "%zu tets in %zu of %zu compartments; %zu recovery Steiner points, %zu welded, %zu degenerate dropped  "
                       "(%.0f ms)\n", cs.plc_vertices, cs.plc_triangles, cs.junction_edges, cs.interior, fill, m.tets.size() / 4,
                       cs.kept_compartments, cs.compartments, cs.steiner, cs.welded, cs.degenerate, cs.ms);

            if (cfg.o.opt) {
                TN_FPRINTF(stderr, "[opt]   %d 3-2 + %d 2-3 flips, %d collapses, %d Steiner points, %d moves\n", os.flips32,
                           os.flips23, os.collapses, os.steiner, os.moves);
            }

            const tn::MeshReport rep = tn::check_mesh(m);
            tn::print_report(rep, "cdt");

            if (!cfg.output.empty()) {
                if (cfg.faces) {
                    std::vector<int32_t> f;
                    tn::extract_faces(m.tets, m.tet_labels, m.nodes, f);

                    for (size_t i = 0; i + 4 < f.size(); i += 5) {
                        m.tris.insert(m.tris.end(), f.begin() + static_cast<std::ptrdiff_t>(i), f.begin() + static_cast<std::ptrdiff_t>(i) + 3);
                        m.tri_labels.push_back(f[i + 3]);
                        m.tri_labels.push_back(f[i + 4]);
                    }
                }

                tn::write_jmesh_auto(cfg.output, m);
                TN_FPRINTF(stderr, "[output] %s written: %zu nodes, %zu tets\n", cfg.output.c_str(), m.nodes.size() / 3,
                           m.tets.size() / 4);
            }

            return rep.ok() ? 0 : 3;
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
            return 1;
        }
    }

    if (cfg.mode == "tessellate" && cfg.image.empty() && cfg.shape.empty()) {   // points -> Delaunay tets
        if (cfg.input.empty()) {
            std::fprintf(stderr, "trussnet: --mode tessellate wants -i POINTS (.xyz .off .jmsh)\n");
            return 2;
        }

        try {
            tn::Mesh m = tn::read_mesh(cfg.input);
            const auto t0 = clk::now();
            tn::tessellate_points(m, cfg.o.gpu);
            TN_FPRINTF(stderr, "[tess]  %zu points -> %zu Delaunay tets%s  (%.0f ms)\n", m.nodes.size() / 3, m.tets.size() / 4,
                       m.node_labels.empty() ? "" : ", labelled from the nodes", ms(t0));
            tn::print_report(tn::check_mesh(m), "tessellated");

            if (!cfg.output.empty()) {
                if (cfg.faces) {
                    std::vector<int32_t> f;
                    tn::extract_faces(m.tets, m.tet_labels, m.nodes, f);

                    for (size_t i = 0; i + 4 < f.size(); i += 5) {
                        m.tris.insert(m.tris.end(), f.begin() + static_cast<std::ptrdiff_t>(i), f.begin() + static_cast<std::ptrdiff_t>(i) + 3);
                        m.tri_labels.push_back(f[i + 3]);
                        m.tri_labels.push_back(f[i + 4]);
                    }
                }

                tn::write_jmesh_auto(cfg.output, m);
                TN_FPRINTF(stderr, "[output] %s written: %zu nodes, %zu tets\n", cfg.output.c_str(), m.nodes.size() / 3,
                           m.tets.size() / 4);
            }

            return 0;
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
            return 1;
        }
    }

    // --mode tessellate --image: the given labelled nodes, meshed against the image
    tn::Nodes given_nodes;
    std::vector<double> given_world;

    if (cfg.mode == "tessellate") {
        try {
            const tn::Mesh pts = tn::read_mesh(cfg.input);
            const size_t n = pts.nodes.size() / 3;

            if (pts.node_labels.size() != n) {
                throw std::runtime_error(cfg.input + ": --image wants labelled nodes (NodeLabel; e.g. from --mode points)");
            }

            given_world = pts.nodes;
            given_nodes.lab.resize(n);
            given_nodes.typ.assign(n, 0);
            given_nodes.part.assign(2 * n, 0xFFFF);
            given_nodes.part3.assign(n, 0xFFFF);

            for (size_t i = 0; i < n; ++i) {
                given_nodes.lab[i] = static_cast<uint16_t>(pts.node_labels[i]);

                if (pts.node_types.size() == n) {
                    given_nodes.typ[i] = static_cast<uint8_t>(pts.node_types[i]);
                }

                if (pts.node_partners.size() == 3 * n) {
                    auto l = [](int32_t x) {
                        return x < 0 ? static_cast<uint16_t>(0xFFFF) : static_cast<uint16_t>(x);
                    };
                    given_nodes.part[2 * i] = l(pts.node_partners[3 * i]);
                    given_nodes.part[2 * i + 1] = l(pts.node_partners[3 * i + 1]);
                    given_nodes.part3[i] = l(pts.node_partners[3 * i + 2]);
                }
            }
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
            return 1;
        }

        cfg.input = cfg.image;   // the volume below is the image (or --shape)
    }

    try {
        tn::LabelVolume lv;
        cfg.o.tpm = cfg.tpm;

        if (cfg.mode == "remesh" || cfg.mode == "repair") {   // surfaces -> soft fields -> the mesher
            if (cfg.input.empty()) {
                throw std::runtime_error("--mode " + cfg.mode + " wants -i SURFACES (.jmsh .bmsh .off .stl)");
            }

            const tn::Mesh surf = tn::read_mesh(cfg.input);
            tn::RasterStats rs;
            tn::remesh_volume(surf, cfg.raster_voxel, cfg.o, lv, rs);
            TN_FPRINTF(stderr, "[remesh] %zu faces (%zu reoriented; %zu exposed faces / parts, the rest buried) -> %d region(s)%s "
                       "on a %d x %d x %d raster of %.4g  (%.0f ms)\n", rs.faces, rs.flipped, rs.boundary_faces, rs.regions,
                       rs.shells ? (" of " + std::to_string(rs.shells) + " shells").c_str() : "", rs.nx, rs.ny, rs.nz, rs.voxel, rs.ms);
            cfg.mode = cfg.mode == "repair" ? "surface" : "mesh";
        } else if (!cfg.input.empty()) {
            const auto tl = clk::now();
            size_t filled = 0;
            lv = tn::load_volume_file(cfg.input, cfg.o, &filled);

            if (!lv.soft_volume.empty()) {
                TN_FPRINTF(stderr, "[tpm]   tissue probabilities -> labels 0..%d (argmax%s); %zu enclosed exterior voxels "
                           "filled  (%.0f ms)\n", lv.maxlabel, cfg.tpm.fields ? ", probability interfaces" : "", filled,
                           ms(tl));
            }
        } else {
            lv = tn::make_shape(cfg.shape, cfg.dim);
        }

        if (lv.nz == 1 && cfg.mode != "mesh") {
            throw std::runtime_error("a single-slice (2-D) input: only --mode mesh");
        }

        if (lv.nz == 1) {   // a single slice: the 2-D mesher (triangles)
            auto given = [&](const char* f) {
                return std::find(cfg.given.begin(), cfg.given.end(), f) != cfg.given.end();
            };
            tn::Image2D im;
            im.nx = lv.nx;
            im.ny = lv.ny;
            im.vs = { { lv.voxelsize[0], lv.voxelsize[1] } };
            im.affine = { { lv.affine[0], lv.affine[1], lv.affine[3], lv.affine[4], lv.affine[5], lv.affine[7] } };

            if (!lv.gray.empty() && (!cfg.o.thresholds.empty() || !lv.thresholds.empty())) {   // gray-scale
                im.gray = lv.gray;
                im.thresholds = cfg.o.thresholds.empty() ? lv.thresholds : cfg.o.thresholds;
            } else {
                im.lab = lv.data;
            }

            tn::Mesh2DOptions o2;
            o2.size = cfg.o.grid.hbase;
            o2.hmin = cfg.o.grid.hmin;
            o2.hmax = cfg.o.grid.hmax;
            o2.hlab = cfg.o.grid.hlab;
            o2.isize = cfg.o.grid.isize;
            o2.gray_sigma = cfg.o.gray_sigma;
            o2.verbose = cfg.o.relax.verbose;

            if (given("--K")) {
                o2.K = cfg.o.grid.K;
            }

            if (given("--grad")) {
                o2.grad = cfg.o.grid.g;
            }

            if (given("--sigma")) {
                o2.sigma = cfg.o.grid.sigma;
            }

            if (given("--iters")) {
                o2.iters = cfg.o.relax.max_iters;
            }

            if (given("--repair")) {
                o2.repair = cfg.o.max_repair;
            }

            if (given("--smooth")) {
                o2.smooth = cfg.o.smooth;
            }

            tn::Mesh2D M;
            tn::Mesh2DStats st;
            tn::mesh2d(im, o2, M, st);
            TN_FPRINTF(stderr, "[2d]    %d x %d pixels -> %zu nodes, %zu triangles (%zu junctions, %d iterations, %zu repairs); "
                       "conformity: %zu bad edges, %zu spanning; min angle %.1f deg, q min %.3f p5 %.3f median %.3f  "
                       "(%.0f ms)\n", im.nx, im.ny, st.nodes, st.tris, st.junctions, st.iterations, st.repairs,
                       st.bad_edges, st.spanning, st.min_angle, st.q_min, st.q_p5, st.q_median, st.ms_total);

            if (!cfg.output.empty()) {
                tn::Mesh out;

                for (size_t v = 0; v < M.node.size() / 2; ++v) {
                    out.nodes.insert(out.nodes.end(), { M.node[2 * v], M.node[2 * v + 1], 0.0 });
                }

                out.tris = M.tri;

                for (int32_t l : M.label) {
                    out.tri_labels.insert(out.tri_labels.end(), { l, l });
                }

                tn::write_jmesh_auto(cfg.output, out);
                TN_FPRINTF(stderr, "[output] %s written (MeshTri: v1 v2 v3 label label)\n", cfg.output.c_str());
            }

            return 0;
        }

        cfg.o.report = true;
        cfg.o.stop_after_relax = cfg.mode == "points";
        cfg.o.surface_only = cfg.mode == "surface" && !cfg.exact_tess;   // (repair is surface by now)

        if (cfg.mode == "tessellate") {   // the given nodes, in this image's grid frame
            tn::world_to_nodes(lv, given_world, given_nodes.P);
            cfg.o.start_nodes = &given_nodes;
        }

        tn::PipelineResult r;
        tn::run_pipeline(lv, cfg.o, r);

        if (!cfg.output.empty()) {   // nodes to world coordinates through the affine
            tn::Mesh out;
            tn::nodes_to_world(lv, r.mesh, out.nodes);
            const auto tw = clk::now();
            std::string what;

            if (cfg.mode == "points") {   // the relaxed nodes and their labels
                const tn::Nodes& nd = r.nodes;
                const size_t n = nd.size();
                out.node_labels.resize(n);
                out.node_types.resize(n);
                out.node_partners.resize(3 * n);

                for (size_t i = 0; i < n; ++i) {
                    auto lab = [](uint16_t l) {
                        return l == 0xFFFF ? -1 : static_cast<int32_t>(l);
                    };
                    out.node_labels[i] = nd.lab[i];
                    out.node_types[i] = nd.typ[i];
                    out.node_partners[3 * i] = nd.typ[i] >= 1 ? lab(nd.part[2 * i]) : -1;
                    out.node_partners[3 * i + 1] = nd.typ[i] >= 2 ? lab(nd.part[2 * i + 1]) : -1;
                    out.node_partners[3 * i + 2] = nd.typ[i] >= 3 && i < nd.part3.size() ? lab(nd.part3[i]) : -1;
                }

                what = "MeshNode + NodeLabel / NodeType / NodePartner";
            } else {
                if (cfg.mode == "mesh" || cfg.mode == "tessellate") {
                    out.tets = r.mesh.tets;
                    out.tet_labels = r.mesh.label;
                    what = "MeshElem";
                }

                if (cfg.mode == "surface" || cfg.faces) {
                    std::vector<int32_t> f;
                    tn::extract_faces(r.mesh.tets, r.mesh.label, out.nodes, f);

                    for (size_t i = 0; i + 4 < f.size(); i += 5) {
                        out.tris.insert(out.tris.end(), f.begin() + static_cast<std::ptrdiff_t>(i),
                                        f.begin() + static_cast<std::ptrdiff_t>(i) + 3);
                        out.tri_labels.push_back(f[i + 3]);
                        out.tri_labels.push_back(f[i + 4]);
                    }

                    what += std::string(what.empty() ? "" : " + ") + "MeshTri (v1 v2 v3 inner outer)";
                }

                if (cfg.mode == "surface") {   // only the surfaces' own nodes
                    tn::compact_nodes(out);
                }
            }

            tn::write_jmesh_auto(cfg.output, out);
            TN_FPRINTF(stderr, "[output] %s written: %zu nodes, %s  (%.0f ms)\n", cfg.output.c_str(), out.nodes.size() / 3,
                       what.c_str(), ms(tw));
        }
    } catch (const std::exception& e) {
        TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
        return 1;
    }

    return 0;
}
