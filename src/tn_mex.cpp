// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_mex.cpp -- MATLAB / GNU Octave MEX gateway (built as trussnet_mex; called by
// matlab/trussnet.m):
//
//   [node, elem, face, info] = trussnet_mex(vol, opt)
//
//   vol   3-D array: integer labels (0 = exterior), or a gray-scale intensity when
//         opt.thresholds is set (label = number of thresholds <= intensity); or a
//         4-D (x, y, z, class) tissue-probability map: labels = the argmax
//         (opt.tpmexterior: 1-based exterior channels, default none -> 1 - sum)
//   opt   struct, fields as tn::set_option (size, hmin, lsize, thresholds, gpu,
//         reratio, ...) plus
//           voxelsize  [dx dy dz] mm (default 1; element sizes are in mm)
//           affine     4x4 voxel(0-based i,j,k) -> world (the voxel size, unless
//                      given, from its columns); default: MATLAB index space
//                      scaled by voxelsize, i.e. voxel (i,j,k) at [i j k] .* voxelsize
//         lsize: a vector (lsize(l) = size of label l, 0 = default) or an N x 2
//         [label size] matrix.
//         sizing: a sizing field in mm with vol's (spatial) size, 0 = automatic at
//         that voxel; or a vector, one size per label (N, or N+1 from label 0) /
//         per threshold level / per TPM channel, 0 = default.
//   node  N x 3 double; elem M x 5 [v1..v4 label] (1-based); face P x 5
//         [v1 v2 v3 inner outer] (1-based; outer = 0 on the exterior surface,
//         normals point from inner to outer; computed only if requested)
//   info  struct: counts, conformity, quality, timings

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "mex.h"
#include "tn_2d.h"
#include "tn_log.h"
#include "tn_pipeline.h"

namespace {

void mex_log(const char* s) {
    mexPrintf("%s", s);
#ifndef OCTAVE_VERSION_HEX
    mexEvalString("drawnow;");   // show progress while the call runs
#endif
}

template <typename T>
void copy_as(const mxArray* a, std::vector<double>& v) {
    const T* p = static_cast<const T*>(mxGetData(a));

    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<double>(p[i]);
    }
}

std::vector<double> to_doubles(const mxArray* a) {
    const size_t n = mxGetNumberOfElements(a);
    std::vector<double> v(n);

    if (mxIsLogical(a)) {
        const mxLogical* p = mxGetLogicals(a);

        for (size_t i = 0; i < n; ++i) {
            v[i] = p[i] ? 1.0 : 0.0;
        }

        return v;
    }

    if (!mxIsNumeric(a)) {
        throw std::runtime_error("expected a numeric or logical value");
    }

    switch (mxGetClassID(a)) {
        case mxDOUBLE_CLASS:
            copy_as<double>(a, v);
            break;

        case mxSINGLE_CLASS:
            copy_as<float>(a, v);
            break;

        case mxINT8_CLASS:
            copy_as<int8_t>(a, v);
            break;

        case mxUINT8_CLASS:
            copy_as<uint8_t>(a, v);
            break;

        case mxINT16_CLASS:
            copy_as<int16_t>(a, v);
            break;

        case mxUINT16_CLASS:
            copy_as<uint16_t>(a, v);
            break;

        case mxINT32_CLASS:
            copy_as<int32_t>(a, v);
            break;

        case mxUINT32_CLASS:
            copy_as<uint32_t>(a, v);
            break;

        case mxINT64_CLASS:
            copy_as<int64_t>(a, v);
            break;

        case mxUINT64_CLASS:
            copy_as<uint64_t>(a, v);
            break;

        default:
            throw std::runtime_error("unsupported numeric class");
    }

    return v;
}

void set_field(mxArray* s, const char* name, double v) {
    mxSetField(s, 0, name, mxCreateDoubleScalar(v));
}

// 2-D: [node, elem, face, info] = trussnet_mex(img2d, opt): node N x 2, elem M x 4
// [v1 v2 v3 label], face P x 4 [v1 v2 inner outer] (1-based), info struct
void mex2d(int nlhs, mxArray* plhs[], const mxArray* V, const mxArray* O) {
    tn::Mesh2DOptions o;
    tn::Image2D im;
    std::vector<double> ps = { 1, 1 }, aff, sizing;
    bool ps_given = false;

    if (O && !mxIsEmpty(O)) {
        if (!mxIsStruct(O)) {
            throw std::runtime_error("opt must be a struct");
        }

        for (int f = 0; f < mxGetNumberOfFields(O); ++f) {
            const std::string name = mxGetFieldNameByNumber(O, f);
            const mxArray* a = mxGetFieldByNumber(O, 0, f);

            if (!a || mxIsEmpty(a)) {
                continue;
            }

            std::string key;

            for (char c : name)
                if (c != '_') {
                    key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }

            if (key == "voxelsize" || key == "pixelsize") {
                ps = to_doubles(a);
                ps_given = true;

                if (ps.size() == 1) {
                    ps.assign(2, ps[0]);
                }

                if (ps.size() < 2) {
                    throw std::runtime_error("opt.pixelsize must have 1 or 2 values");
                }

                ps.resize(2);
            } else if (key == "affine") {   // 2 x 3 or 3 x 3, pixel (0-based i, j) -> world
                if (mxGetN(a) != 3 || (mxGetM(a) != 2 && mxGetM(a) != 3)) {
                    throw std::runtime_error("a 2-D opt.affine must be 2 x 3 or 3 x 3");
                }

                const std::vector<double> v = to_doubles(a);   // column-major
                const size_t m = mxGetM(a);
                aff.resize(6);

                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 3; ++c) {
                        aff[3 * r + c] = v[c * m + r];
                    }
            } else if (key == "sizing") {
                sizing = to_doubles(a);
            } else if (key == "lsize") {
                std::vector<double> v = to_doubles(a), pairs;

                if (mxGetN(a) == 2 && mxGetM(a) > 1) {
                    const size_t m = mxGetM(a);

                    for (size_t r = 0; r < m; ++r) {
                        pairs.push_back(v[r]);
                        pairs.push_back(v[m + r]);
                    }
                } else {
                    for (size_t l = 0; l < v.size(); ++l)
                        if (v[l] > 0) {
                            pairs.push_back(static_cast<double>(l + 1));
                            pairs.push_back(v[l]);
                        }
                }

                tn::set_option2d(o, im.thresholds, "lsize", pairs);
            } else if (mxIsChar(a) || !tn::set_option2d(o, im.thresholds, key, to_doubles(a))) {
                mexWarnMsgIdAndTxt("trussnet:opt", "unknown 2-D option '%s' ignored", name.c_str());
            }
        }
    }

    im.nx = static_cast<int>(mxGetM(V));
    im.ny = static_cast<int>(mxGetN(V));
    const std::vector<double> d = to_doubles(V);

    if (!im.thresholds.empty()) {
        im.gray.assign(d.begin(), d.end());
    } else {
        im.lab.resize(d.size());

        for (size_t i = 0; i < d.size(); ++i) {
            const double x = std::round(d[i]);

            if (!(x >= 0 && x <= 65535)) {
                throw std::runtime_error("labels must be integers in 0..65535 (for a gray-scale image set opt.thresholds)");
            }

            im.lab[i] = static_cast<uint16_t>(x);
        }
    }

    if (!aff.empty()) {
        for (int k = 0; k < 6; ++k) {
            im.affine[k] = aff[k];
        }

        if (!ps_given)
            for (int c = 0; c < 2; ++c) {
                ps[c] = std::hypot(im.affine[c], im.affine[3 + c]);
            }
    } else {   // MATLAB index space (1-based) scaled by the pixel size
        im.affine = { { ps[0], 0, ps[0], 0, ps[1], ps[1] } };
    }

    im.vs = { { ps[0], ps[1] } };

    if (!sizing.empty()) {
        if (sizing.size() == d.size()) {
            o.hvox.assign(sizing.begin(), sizing.end());
        } else {
            int nl = 0;

            if (!im.thresholds.empty()) {
                nl = static_cast<int>(im.thresholds.size());
            } else {
                for (uint16_t l : im.lab) {
                    nl = std::max<int>(nl, l);
                }
            }

            if (sizing.size() != static_cast<size_t>(nl) && sizing.size() != static_cast<size_t>(nl) + 1) {
                throw std::runtime_error("sizing: " + std::to_string(sizing.size()) + " values; want one per pixel or per label (" +
                                         std::to_string(nl) + " or " + std::to_string(nl + 1) + ")");
            }

            const int off = sizing.size() == static_cast<size_t>(nl) ? 1 : 0;

            for (size_t k = 0; k < sizing.size(); ++k) {
                const int l = static_cast<int>(k) + off;

                if (sizing[k] > 0 && l > 0) {
                    if (static_cast<int>(o.hlab.size()) <= l) {
                        o.hlab.resize(l + 1, 0.0f);
                    }

                    o.hlab[l] = static_cast<float>(sizing[k]);
                }
            }
        }
    }

    tn::Mesh2D M;
    tn::Mesh2DStats st;
    tn::mesh2d(im, o, M, st);
    const size_t nn = M.node.size() / 2, nt = M.label.size(), ne = M.edge.size() / 4;
    plhs[0] = mxCreateDoubleMatrix(nn, 2, mxREAL);
    double* pn = mxGetPr(plhs[0]);

    for (size_t i = 0; i < nn; ++i) {
        pn[i] = M.node[2 * i];
        pn[nn + i] = M.node[2 * i + 1];
    }

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(nt, 4, mxREAL);
        double* pe = mxGetPr(plhs[1]);

        for (size_t t = 0; t < nt; ++t) {
            for (int k = 0; k < 3; ++k) {
                pe[k * nt + t] = M.tri[3 * t + k] + 1.0;
            }

            pe[3 * nt + t] = M.label[t];
        }
    }

    if (nlhs > 2) {
        plhs[2] = mxCreateDoubleMatrix(ne, 4, mxREAL);
        double* pf = mxGetPr(plhs[2]);

        for (size_t e = 0; e < ne; ++e)
            for (int k = 0; k < 4; ++k) {
                pf[k * ne + e] = M.edge[4 * e + k] + (k < 2 ? 1.0 : 0.0);
            }
    }

    if (nlhs > 3) {
        const char* fn[] = { "nodes", "tris", "seeds", "junctions", "iterations", "repairrounds", "repairs", "badedges",
                             "spanning", "minangle", "qmin", "qp5", "qmedian", "ms_fields", "ms_relax", "ms_mesh",
                             "ms_total"
                           };
        const int nfn = sizeof(fn) / sizeof(fn[0]);
        plhs[3] = mxCreateStructMatrix(1, 1, nfn, fn);
        const double vals[] = { double(nn), double(nt), double(st.seeds), double(st.junctions), double(st.iterations),
                                double(st.repair_rounds), double(st.repairs), double(st.bad_edges), double(st.spanning),
                                st.min_angle, st.q_min, st.q_p5, st.q_median, st.ms_fields, st.ms_relax, st.ms_mesh,
                                st.ms_total
                              };

        for (int k = 0; k < nfn; ++k) {
            set_field(plhs[3], fn[k], vals[k]);
        }

        mxArray* la = mxCreateDoubleMatrix(1, st.label_area.size(), mxREAL);
        mxArray* lp = mxCreateDoubleMatrix(1, st.label_pixels.size(), mxREAL);
        std::copy(st.label_area.begin(), st.label_area.end(), mxGetPr(la));
        std::copy(st.label_pixels.begin(), st.label_pixels.end(), mxGetPr(lp));
        mxAddField(plhs[3], "labelarea");
        mxAddField(plhs[3], "labelpixels");
        mxSetField(plhs[3], 0, "labelarea", la);
        mxSetField(plhs[3], 0, "labelpixels", lp);
    }
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("trussnet:args", "usage: [node, elem, face, info] = trussnet_mex(vol, opt)");
    }

    try {
        const mxArray* V = prhs[0];

        if (!mxIsChar(V) && mxGetNumberOfDimensions(V) == 2 && mxGetM(V) > 1 && mxGetN(V) > 1) {   // a 2-D image
            tn::set_log_writer(mex_log);
            mex2d(nlhs, plhs, V, nrhs > 1 ? prhs[1] : nullptr);
            tn::set_log_writer(nullptr);
            return;
        }

        const bool from_file = mxIsChar(V);   // a volume file name: its own grid and affine
        const mwSize ndv = from_file ? 3 : mxGetNumberOfDimensions(V);

        if (!from_file && ndv != 3 && ndv != 4) {
            throw std::runtime_error("vol must be a 3-D array, or 4-D (x, y, z, class) tissue probabilities");
        }

        tn::PipelineOptions o;
        std::vector<double> vs = { 1, 1, 1 }, aff, sizing;
        bool vs_given = false;

        if (nrhs > 1 && !mxIsEmpty(prhs[1])) {
            const mxArray* O = prhs[1];

            if (!mxIsStruct(O)) {
                throw std::runtime_error("opt must be a struct");
            }

            for (int f = 0; f < mxGetNumberOfFields(O); ++f) {
                const std::string name = mxGetFieldNameByNumber(O, f);
                const mxArray* a = mxGetFieldByNumber(O, 0, f);

                if (!a || mxIsEmpty(a)) {
                    continue;
                }

                std::string key;

                for (char c : name) {
                    if (c != '_') {
                        key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    }
                }

                if (key == "voxelsize") {
                    vs_given = true;
                    vs = to_doubles(a);

                    if (vs.size() == 1) {
                        vs.assign(3, vs[0]);
                    }

                    if (vs.size() != 3) {
                        throw std::runtime_error("opt.voxelsize must have 1 or 3 values");
                    }
                } else if (key == "affine") {
                    if (mxGetM(a) != 4 || mxGetN(a) != 4) {
                        throw std::runtime_error("opt.affine must be 4x4");
                    }

                    aff = to_doubles(a);   // column-major
                } else if (key == "lsize") {
                    std::vector<double> v = to_doubles(a), pairs;

                    if (mxGetN(a) == 2 && mxGetM(a) > 1) {   // N x 2 [label size]
                        const size_t m = mxGetM(a);

                        for (size_t r = 0; r < m; ++r) {
                            pairs.push_back(v[r]);
                            pairs.push_back(v[m + r]);
                        }
                    } else {   // lsize(l): label l
                        for (size_t l = 0; l < v.size(); ++l)
                            if (v[l] > 0) {
                                pairs.push_back(static_cast<double>(l + 1));
                                pairs.push_back(v[l]);
                            }
                    }

                    tn::set_option(o, "lsize", pairs);
                } else if (key == "sizing") {   // a sizing field (vol's grid) or one per label / channel
                    sizing = to_doubles(a);
                } else if (key == "tpmexterior") {   // 1-based channels in MATLAB
                    std::vector<double> c = to_doubles(a);

                    for (double& x : c) {
                        x -= 1.0;
                    }

                    tn::set_option(o, key, c);
                } else if (mxIsChar(a)) {
                    char* s = mxArrayToString(a);
                    const std::string str = s ? s : "";
                    mxFree(s);

                    if (!tn::set_option(o, key, {}, str)) {
                        mexWarnMsgIdAndTxt("trussnet:opt", "unknown option '%s' ignored", name.c_str());
                    }
                } else if (!tn::set_option(o, key, to_doubles(a))) {
                    mexWarnMsgIdAndTxt("trussnet:opt", "unknown option '%s' ignored", name.c_str());
                }
            }
        }

        tn::LabelVolume lv;
        size_t tpm_filled = 0;
        std::vector<int> tpm_map;

        if (from_file) {   // labels / gray-scale / TPM through the C++ loaders
            char* fs = mxArrayToString(V);
            const std::string path = fs ? fs : "";
            mxFree(fs);
            lv = tn::load_volume_file(path, o, &tpm_filled, &tpm_map);
        } else {
            // the array (MATLAB is column-major: x fastest, as tn::LabelVolume; a
            // 4-D array is then channel-major, as tn::Tpm)
            const mwSize* dims = mxGetDimensions(V);
            lv.nx = static_cast<int>(dims[0]);
            lv.ny = static_cast<int>(dims[1]);
            lv.nz = static_cast<int>(dims[2]);
            const std::vector<double> d = to_doubles(V);
            lv.data.assign(ndv == 4 ? d.size() / dims[3] : d.size(), 0);

            if (ndv == 4) {   // tissue probabilities
                tn::Tpm t;
                t.nx = lv.nx;
                t.ny = lv.ny;
                t.nz = lv.nz;
                t.C = static_cast<int>(dims[3]);
                t.p.assign(d.begin(), d.end());
                tpm_map = tn::apply_tpm(t, o.tpm, lv, &tpm_filled);
            } else if (!o.thresholds.empty()) {
                lv.gray.assign(d.begin(), d.end());
            } else {
                for (size_t i = 0; i < d.size(); ++i) {
                    const double x = std::round(d[i]);

                    if (!(x >= 0 && x <= 65535)) {
                        throw std::runtime_error("labels must be integers in 0..65535 (for a gray-scale volume set "
                                                 "opt.thresholds)");
                    }

                    lv.data[i] = static_cast<uint16_t>(x);
                }
            }

            if (!aff.empty()) {   // column-major 4x4 -> row-major
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) {
                        lv.affine[4 * r + c] = aff[4 * c + r];
                    }

                if (!vs_given) {   // the voxel size from the affine's columns (as the Python binding)
                    for (int c = 0; c < 3; ++c) {
                        vs[c] = std::sqrt(lv.affine[c] * lv.affine[c] + lv.affine[4 + c] * lv.affine[4 + c] +
                                          lv.affine[8 + c] * lv.affine[8 + c]);
                    }
                }
            } else {   // MATLAB index space (1-based) scaled by the voxel size
                lv.affine = { { vs[0], 0, 0, vs[0], 0, vs[1], 0, vs[1], 0, 0, vs[2], vs[2], 0, 0, 0, 1 } };
            }

            lv.voxelsize = { { vs[0], vs[1], vs[2] } };
        }

        tn::apply_user_sizing(lv, sizing, tpm_map, o);
        tn::set_log_writer(mex_log);
        tn::PipelineResult r;
        tn::run_pipeline(lv, o, r);
        std::vector<double> nodes;
        tn::nodes_to_world(lv, r.mesh, nodes);
        const size_t nn = nodes.size() / 3, ne = r.mesh.tets.size() / 4;

        plhs[0] = mxCreateDoubleMatrix(nn, 3, mxREAL);
        double* pn = mxGetPr(plhs[0]);

        for (size_t i = 0; i < nn; ++i)
            for (int k = 0; k < 3; ++k) {
                pn[k * nn + i] = nodes[3 * i + k];
            }

        if (nlhs > 1) {
            plhs[1] = mxCreateDoubleMatrix(ne, 5, mxREAL);
            double* pe = mxGetPr(plhs[1]);

            for (size_t i = 0; i < ne; ++i) {
                for (int k = 0; k < 4; ++k) {
                    pe[k * ne + i] = r.mesh.tets[4 * i + k] + 1.0;
                }

                pe[4 * ne + i] = r.mesh.label[i];
            }
        }

        if (nlhs > 2) {
            std::vector<int32_t> faces;
            tn::extract_faces(r.mesh.tets, r.mesh.label, nodes, faces);
            const size_t nf = faces.size() / 5;
            plhs[2] = mxCreateDoubleMatrix(nf, 5, mxREAL);
            double* pf = mxGetPr(plhs[2]);

            for (size_t i = 0; i < nf; ++i)
                for (int k = 0; k < 5; ++k) {
                    pf[k * nf + i] = faces[5 * i + k] + (k < 3 ? 1.0 : 0.0);
                }
        }

        if (nlhs > 3) {
            const tn::TetStats& t = r.tess;
            const char* fn[] = { "nodes", "tets", "seeds", "iterations", "repairrounds", "badfaces", "badedges",
                                 "spanning", "mindihedral", "slivers10", "joeliumin", "joeliup5", "joeliumedian",
                                 "volume", "usedgpu", "ms_grid", "ms_seed", "ms_relax", "ms_tess", "ms_total",
                                 "tpmfilled"
                               };
            const int nfn = sizeof(fn) / sizeof(fn[0]);
            plhs[3] = mxCreateStructMatrix(1, 1, nfn, fn);
            const double vals[] = { double(nn), double(ne), double(r.seeds), double(r.relax.iters),
                                    double(t.repair_rounds), double(t.bad_faces), double(t.bad_edges),
                                    double(t.bad_span), t.min_dihedral, double(t.slivers10), t.joe_liu_min,
                                    t.joe_liu_p5, t.joe_liu_med, t.volume, r.used_gpu ? 1.0 : 0.0, r.ms_grid,
                                    r.ms_seed, r.ms_relax, r.ms_tess, r.ms_total, double(tpm_filled)
                                  };

            for (int k = 0; k < nfn; ++k) {
                set_field(plhs[3], fn[k], vals[k]);
            }
        }

        tn::set_log_writer(nullptr);
    } catch (const std::exception& e) {
        tn::set_log_writer(nullptr);
        mexErrMsgIdAndTxt("trussnet:error", "trussnet: %s", e.what());
    }
}
