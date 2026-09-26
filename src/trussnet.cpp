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
#include "tn_log.h"
#include "tn_2d.h"
#include "tn_pipeline.h"
#include "tn_tpm.h"
#include "tn_shapes.h"
#include "tn_volume.h"

namespace {

struct Config {
    std::string input, shape, output;
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
                 "  --dt T           Jacobi relaxation factor (default 0.5)\n"
                 "  --snap S         interior nodes within S*h of an interface join it (default 0.5)\n"
                 "  --lsize L:H,..   per-label element size (mm), e.g. 1:4,2:3 (default --size)\n"
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

}  // namespace

int main(int argc, char** argv) {
    Config cfg;

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

    if (cfg.input.empty() && cfg.shape.empty()) {
        usage(argv[0]);
        return 2;
    }

    typedef std::chrono::steady_clock clk;
    auto ms = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
    };

    try {
        tn::LabelVolume lv;
        cfg.o.tpm = cfg.tpm;

        if (!cfg.input.empty()) {
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
        tn::PipelineResult r;
        tn::run_pipeline(lv, cfg.o, r);

        if (!cfg.output.empty()) {   // nodes to world coordinates through the affine
            tn::Mesh out;
            tn::nodes_to_world(lv, r.mesh, out.nodes);
            out.tets = r.mesh.tets;
            out.tet_labels = r.mesh.label;
            const auto tw = clk::now();
            tn::write_jmesh_auto(cfg.output, out);
            TN_FPRINTF(stderr, "[output] %s written (%.0f ms)\n", cfg.output.c_str(), ms(tw));
        }
    } catch (const std::exception& e) {
        TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
        return 1;
    }

    return 0;
}
