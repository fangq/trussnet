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

#include "nlohmann/json.hpp"
#include "tn_grid.h"
#include "tn_particles.h"
#ifdef TN_HAS_OPENCL
    #include "tn_gpu.h"
#endif
#include "tn_tetra.h"
#include "tn_jmesh.h"
#include "tn_mesh.h"
#include "tn_log.h"
#include "tn_shapes.h"
#include "tn_volume.h"

namespace {

struct Config {
    std::string input, shape, output, dump_grid, dump_nodes;
    int dim = 96;
    tn::GridParams grid;
    tn::RelaxParams relax;
    int max_repair = 6;
    int gpu = -2;   // -2: CPU (OpenMP); else the OpenCL device (-1 = first GPU)
    std::vector<float> thresholds;   // gray-scale input: iso-values
    float gray_sigma = 0.0f;
};

void usage(const char* exe) {
    std::fprintf(stderr,
                 "trussnet -- GPU particle (truss) multi-label tetrahedral mesher\n"
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
                 "  --thresholds T1,T2,..  gray-scale input: label = number of thresholds <= intensity;\n"
                 "                   the interfaces are the iso-surfaces (0 = below T1 = exterior)\n"
                 "  --gray-sigma S   Gaussian pre-smoothing of the gray-scale input (voxels)\n"
                 "  --gpu [N]        relax on OpenCL device N (default: the first GPU)\n"
                 "  --jseed C        junction-line seeds, one per C x spacing cell (default 0.8, 0 = off)\n"
                 "  --no-corners     no fixed nodes where >= 4 labels meet\n"
                 "  --trap M         boundary trapping: smooth (sub-voxel interface, default) or\n"
                 "                   voxel (exact voxel faces: DDA walk + nearest staircase face)\n"
                 "  --repair N       max restricted-Delaunay repair rounds (default 6)\n"
                 "  --dump-grid F    write labels / h / grade to a BJData .bnii (debug)\n"
                 "  --dump-nodes F   write the relaxed nodes (+label, type) to BJData (debug)\n"
                 "  -v               progress\n"
                 "shapes:", exe);

    for (const std::string& s : tn::shape_names()) {
        std::fprintf(stderr, " %s", s.c_str());
    }

    std::fprintf(stderr, "\n");
}

// a minimal JNIfTI (BJData) volume writer for debugging: labels, h, grade
void dump_grid(const std::string& path, const tn::LabelVolume& lv, const tn::Grid& g) {
    using nlohmann::json;
    auto arr = [&](const char* type, const json& data) {
        return json{ { "_ArrayType_", type }, { "_ArraySize_", { g.nz, g.ny, g.nx } }, { "_ArrayData_", data } };
    };
    json j;
    j["NIFTIHeader"] = { { "Dim", { g.nx, g.ny, g.nz } }, { "VoxelSize", { g.vs[0], g.vs[1], g.vs[2] } },
        { "hmin", g.hmin }, { "hmax", g.hmax } };
    j["Labels"] = arr("uint16", lv.data);
    j["Size"] = arr("single", g.h);
    j["Grade"] = arr("uint8", g.grade);
    std::vector<std::uint8_t> out = json::to_bjdata(j, true, true);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

// debug dump: one JSON header line {name: [dtype, shape, byte offset]}, then the
// raw little-endian arrays (read with tools/tnview.py)
void dump_nodes(const std::string& path, const tn::Nodes& nd) {
    using nlohmann::json;
    const int n = static_cast<int>(nd.size());
    const size_t oP = 0, oL = oP + nd.P.size() * 4, oT = oL + nd.lab.size() * 2, oQ = oT + nd.typ.size();
    json j = { { "P", { "float32", { n, 3 }, oP } }, { "Label", { "uint16", { n }, oL } },
        { "Type", { "uint8", { n }, oT } }, { "Partner", { "uint16", { n, 2 }, oQ } }
    };
    std::ofstream f(path, std::ios::binary);
    f << j.dump() << "\n";
    f.write(reinterpret_cast<const char*>(nd.P.data()), static_cast<std::streamsize>(nd.P.size() * 4));
    f.write(reinterpret_cast<const char*>(nd.lab.data()), static_cast<std::streamsize>(nd.lab.size() * 2));
    f.write(reinterpret_cast<const char*>(nd.typ.data()), static_cast<std::streamsize>(nd.typ.size()));
    f.write(reinterpret_cast<const char*>(nd.part.data()), static_cast<std::streamsize>(nd.part.size() * 2));
}

double label_volume(const tn::LabelVolume& lv) {
    size_t c = 0;

    for (uint16_t l : lv.data) {
        c += l != 0;
    }

    return c * lv.voxelsize[0] * lv.voxelsize[1] * lv.voxelsize[2];
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "trussnet: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
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
            cfg.grid.hbase = static_cast<float>(std::atof(next()));
        } else if (a == "--hmin") {
            cfg.grid.hmin = static_cast<float>(std::atof(next()));
        } else if (a == "--hmax") {
            cfg.grid.hmax = static_cast<float>(std::atof(next()));
        } else if (a == "--K") {
            cfg.grid.K = static_cast<float>(std::atof(next()));
        } else if (a == "--grad") {
            cfg.grid.g = static_cast<float>(std::atof(next()));
        } else if (a == "--sigma") {
            cfg.grid.sigma = static_cast<float>(std::atof(next()));
        } else if (a == "--dump-grid") {
            cfg.dump_grid = next();
        } else if (a == "--dump-nodes") {
            cfg.dump_nodes = next();
        } else if (a == "--nseed") {
            cfg.relax.nseed = std::atoi(next());
        } else if (a == "--iters") {
            cfg.relax.max_iters = std::atoi(next());
        } else if (a == "--fscale") {
            cfg.relax.fscale = static_cast<float>(std::atof(next()));
        } else if (a == "--dt") {
            cfg.relax.dt = static_cast<float>(std::atof(next()));
        } else if (a == "--fsurf") {
            cfg.relax.fsurf = static_cast<float>(std::atof(next()));
        } else if (a == "--sigma-thin") {
            cfg.grid.sigma_thin = static_cast<float>(std::atof(next()));
        } else if (a == "--thick") {
            cfg.grid.thick = static_cast<float>(std::atof(next()));
        } else if (a == "--thin-floor") {
            cfg.grid.thin_floor = static_cast<float>(std::atof(next()));
        } else if (a == "--preserve") {
            cfg.grid.preserve = static_cast<float>(std::atof(next()));
        } else if (a == "--thresholds") {   // gray-scale iso-values: t1,t2,...
            cfg.thresholds.clear();
            std::string v = next();
            size_t p0 = 0;

            while (p0 <= v.size()) {
                const size_t p1 = v.find(',', p0);
                cfg.thresholds.push_back(static_cast<float>(std::atof(v.substr(p0, p1 - p0).c_str())));

                if (p1 == std::string::npos) {
                    break;
                }

                p0 = p1 + 1;
            }
        } else if (a == "--gray-sigma") {
            cfg.gray_sigma = static_cast<float>(std::atof(next()));
        } else if (a == "--gpu") {   // optional device index
            cfg.gpu = -1;

            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.gpu = std::atoi(argv[++i]);
            }
        } else if (a == "--jseed") {
            cfg.relax.jseed = static_cast<float>(std::atof(next()));
        } else if (a == "--no-corners") {
            cfg.relax.corners = false;
        } else if (a == "--trap") {
            const std::string m = next();

            if (m != "smooth" && m != "voxel") {
                throw std::runtime_error("--trap wants smooth or voxel");
            }

            cfg.relax.voxel_trap = m == "voxel";
        } else if (a == "--snap") {
            cfg.relax.snap = static_cast<float>(std::atof(next()));
        } else if (a == "--repair") {
            cfg.max_repair = std::atoi(next());
        } else if (a == "-v") {
            cfg.relax.verbose = true;
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
        clk::time_point t0 = clk::now();
        tn::LabelVolume lv = cfg.input.empty() ? tn::make_shape(cfg.shape, cfg.dim)
                                               : tn::load_label_volume(cfg.input, !cfg.thresholds.empty());

        if (!cfg.thresholds.empty()) {   // gray-scale: (re)label by the iso-values
            if (lv.gray.empty()) {
                throw std::runtime_error("--thresholds needs a gray-scale input (or a gray* shape)");
            }

            tn::apply_thresholds(lv, cfg.thresholds, cfg.gray_sigma);
        }
        TN_FPRINTF(stderr, "[input] %d x %d x %d voxels (%.3g x %.3g x %.3g mm), labels 0..%d  (%.0f ms)\n", lv.nx,
                   lv.ny, lv.nz, lv.voxelsize[0], lv.voxelsize[1], lv.voxelsize[2], lv.maxlabel, ms(t0));

        clk::time_point t1 = clk::now();
        tn::Grid g;
        tn::build_grid_cpu(lv, cfg.grid, g);
        TN_FPRINTF(stderr, "[grid]  %zu slots over %d/%d bricks (%d overflow), h in [%.3g, %.3g] mm, %d limit sweeps "
                   "(%.0f ms)\n", g.slot_brick.size(),
                   static_cast<int>(std::count_if(g.bl_slot.begin(), g.bl_slot.end(), [](int s) {
                       return s >= 0;
                   })), g.nbx * g.nby * g.nbz, g.overflow_bricks, g.hmin, g.hmax, g.limit_sweeps, ms(t1));

        if (!cfg.dump_grid.empty()) {
            dump_grid(cfg.dump_grid, lv, g);
        }

        clk::time_point t2 = clk::now();
        tn::Nodes nd;
        tn::seed_cpu(g, cfg.relax, nd);
        TN_FPRINTF(stderr, "[seed]  %zu nodes (%.0f ms)\n", nd.size(), ms(t2));

        if (!cfg.dump_nodes.empty()) {
            dump_nodes(cfg.dump_nodes + ".seed.bjd", nd);
        }

        clk::time_point t3 = clk::now();
        tn::RelaxStats rs;
#ifdef TN_HAS_OPENCL
        if (cfg.gpu > -2) {
            tn::relax_cl(g, cfg.relax, nd, rs, cfg.gpu);
        } else
#endif
        {
            tn::relax_cpu(g, cfg.relax, nd, rs);
        }
        TN_FPRINTF(stderr, "[relax] %d iterations, %d rebuilds, last max move %.3g h (p99 < %.2g h); %zu interior, %zu interface, %zu "
                   "junction, %zu corner  (%.0f ms: hash %.0f, force %.0f, move %.0f)\n", rs.iters, rs.rebuilds,
                   rs.last_move, rs.last_p99, rs.n_interior, rs.n_interface, rs.n_junction, rs.n_corner, ms(t3), rs.ms_hash,
                   rs.ms_force, rs.ms_move);

        if (!cfg.dump_nodes.empty()) {
            dump_nodes(cfg.dump_nodes, nd);
        }

        clk::time_point t4 = clk::now();
        tn::TetOut tm;
        tn::TetStats ts;
        tn::tessellate(g, nd, cfg.relax.voxel_trap, cfg.max_repair, tm, ts);
        TN_FPRINTF(stderr, "[tess]  %zu Delaunay tets -> %zu kept (%zu peeled); conformity: %zu bad faces, %zu edges through label 0, "
                   "%zu spanning; %d repair rounds, %zu repairs  (%.0f ms: delaunay %.0f, label %.0f, check %.0f)\n",
                   ts.delaunay_tets, ts.kept, ts.peeled,
                   ts.bad_faces, ts.bad_edges, ts.bad_span, ts.repair_rounds, ts.repaired, ms(t4), ts.ms_delaunay, ts.ms_label, ts.ms_check);
        TN_FPRINTF(stderr, "[conf]  offending-node distance to its voxel interface (voxels, p50/p95/p99/max, 5 = none within 4): faces "
                   "%.2f/%.2f/%.2f/%.2f, spanning %.2f/%.2f/%.2f/%.2f\n", ts.dev_face[0], ts.dev_face[1], ts.dev_face[2],
                   ts.dev_face[3], ts.dev_span[0], ts.dev_span[1], ts.dev_span[2], ts.dev_span[3]);
        {
            std::string lv_s;
            double worst = 0.0;

            for (size_t l = 1; l < ts.label_vox.size(); ++l)
                if (ts.label_vox[l] > 0) {
                    const double e = 100.0 * (ts.label_vol[l] - ts.label_vox[l]) / ts.label_vox[l];
                    char b2[48];
                    std::snprintf(b2, sizeof(b2), " %zu:%+.1f%%", l, e);
                    lv_s += b2;
                    worst = std::max(worst, std::fabs(e));
                }

            TN_FPRINTF(stderr, "[conf]  per-label volume error (max |%.2f%%|):%s\n", worst, lv_s.c_str());
        }
        TN_FPRINTF(stderr, "[qual]  min dihedral %.2f deg, slivers <10: %zu (%.2f%%) <5: %zu; Joe-Liu min %.3f p5 %.3f "
                   "median %.3f; volume %.1f mm^3 (label volume %.1f)\n", ts.min_dihedral, ts.slivers10,
                   100.0 * ts.slivers10 / std::max<size_t>(1, ts.kept), ts.slivers5, ts.joe_liu_min, ts.joe_liu_p5,
                   ts.joe_liu_med, ts.volume, label_volume(lv));

        if (!cfg.output.empty()) {   // nodes to world coordinates through the affine
            tn::Mesh out;
            const size_t nn = tm.P.size() / 3;
            out.nodes.resize(nn * 3);

            for (size_t i = 0; i < nn; ++i) {
                const double u = tm.P[3 * i] / lv.voxelsize[0], v = tm.P[3 * i + 1] / lv.voxelsize[1],
                             w = tm.P[3 * i + 2] / lv.voxelsize[2];

                for (int r = 0; r < 3; ++r) {
                    out.nodes[3 * i + r] = lv.affine[4 * r] * u + lv.affine[4 * r + 1] * v + lv.affine[4 * r + 2] * w +
                                           lv.affine[4 * r + 3];
                }
            }

            out.tets = tm.tets;
            out.tet_labels = tm.label;
            tn::write_jmesh_auto(cfg.output, out);
        }
    } catch (const std::exception& e) {
        TN_FPRINTF(stderr, "trussnet: fatal: %s\n", e.what());
        return 1;
    }

    return 0;
}
