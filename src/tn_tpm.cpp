// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_tpm.cpp -- see tn_tpm.h. The JNIfTI part follows gpu_brain2mesh's b2m_tpm
// (JData annotated arrays, zlib / base64 through zmat); the NIfTI-1 part reads a
// 4-D .nii[.gz] directly (the siamize reader is 3-D only). The volume stays in
// its file's voxel order; the affine maps it to world coordinates.

#include "tn_tpm.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "siam.h"   // SIAM18_TO_SPM6
#include "zmat.h"   // declarations only; the implementation lives in nifti_io.cpp

namespace tn {

namespace {

using json = nlohmann::json;

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }

    const std::streamsize sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> b(static_cast<size_t>(sz));

    if (sz > 0 && !f.read(reinterpret_cast<char*>(b.data()), sz)) {
        throw std::runtime_error("read failed: " + path);
    }

    return b;
}

std::vector<uint8_t> zdec(const uint8_t* in, size_t n, int zipid) {
    unsigned char* out = nullptr;
    size_t outlen = 0;
    int zret = 0;
    const int rc = zmat_run(n, const_cast<unsigned char*>(in), &outlen, &out, zipid, &zret, 0);

    if (rc != 0 || !out) {
        if (out) {
            zmat_free(&out);
        }

        throw std::runtime_error("zmat decode failed (zipid " + std::to_string(zipid) + ")");
    }

    std::vector<uint8_t> r(out, out + outlen);
    zmat_free(&out);
    return r;
}

bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    return s;
}

// a numeric JNIfTI field, plain (nested) array or a JData annotated array
std::vector<double> numbers(const json& v) {
    std::vector<double> out;

    if (v.is_array()) {
        for (const json& e : v) {
            const std::vector<double> sub = numbers(e);
            out.insert(out.end(), sub.begin(), sub.end());
        }
    } else if (v.is_number()) {
        out.push_back(v.get<double>());
    } else if (v.is_object() && v.contains("_ArrayData_") && v["_ArrayData_"].is_array()) {
        out = numbers(v["_ArrayData_"]);
    }

    return out;
}

// the bytes of a JData byte payload: base64 text, binary, or a list of numbers
std::vector<uint8_t> payload_bytes(const json& z) {
    std::vector<uint8_t> b;

    if (z.is_string()) {
        const std::string s = z.get<std::string>();
        b = zdec(reinterpret_cast<const uint8_t*>(s.data()), s.size(), zmBase64);
    } else if (z.is_binary()) {
        b.assign(z.get_binary().begin(), z.get_binary().end());
    } else if (z.is_array()) {
        b.reserve(z.size());

        for (const json& e : z) {
            b.push_back(static_cast<uint8_t>(e.get<int>()));
        }
    }

    return b;
}

// the bytes of a JData annotated array (zlib / base64 / raw binary / number list)
std::vector<uint8_t> array_bytes(const json& a) {
    if (a.contains("_ArrayZipData_")) {
        const std::string zt = a.value("_ArrayZipType_", std::string("zlib"));
        const int id = zt == "gzip" ? zmGzip : zmZlib;

        if (zt != "zlib" && zt != "gzip") {
            throw std::runtime_error("unsupported _ArrayZipType_ " + zt);
        }

        const std::vector<uint8_t> z = payload_bytes(a["_ArrayZipData_"]);
        return zdec(z.data(), z.size(), id);
    }

    if (a.contains("_ArrayData_")) {
        const json& d = a["_ArrayData_"];

        if (d.is_binary()) {
            return std::vector<uint8_t>(d.get_binary().begin(), d.get_binary().end());
        }

        if (d.is_array()) {   // plain numbers: re-encode as doubles
            std::vector<double> v = numbers(d);
            std::vector<uint8_t> b(v.size() * 8);
            std::memcpy(b.data(), v.data(), b.size());
            return b;
        }
    }

    throw std::runtime_error("NIFTIData: no _ArrayZipData_ / _ArrayData_");
}

// element i of a typed buffer as float
struct Typed {
    const uint8_t* b;
    int code;   // 0 f32 1 f64 2 u8 3 i8 4 u16 5 i16 6 u32 7 i32
    float at(size_t i) const {
        switch (code) {
            case 0: {
                float x;
                std::memcpy(&x, b + 4 * i, 4);
                return x;
            }

            case 1: {
                double x;
                std::memcpy(&x, b + 8 * i, 8);
                return static_cast<float>(x);
            }

            case 2:
                return b[i];

            case 3:
                return static_cast<int8_t>(b[i]);

            case 4: {
                uint16_t x;
                std::memcpy(&x, b + 2 * i, 2);
                return x;
            }

            case 5: {
                int16_t x;
                std::memcpy(&x, b + 2 * i, 2);
                return x;
            }

            case 6: {
                uint32_t x;
                std::memcpy(&x, b + 4 * i, 4);
                return static_cast<float>(x);
            }

            default: {
                int32_t x;
                std::memcpy(&x, b + 4 * i, 4);
                return static_cast<float>(x);
            }
        }
    }
};

const int kElemSize[8] = { 4, 8, 1, 1, 2, 2, 4, 4 };

int jdata_type(const std::string& t) {
    const char* names[8] = { "single", "double", "uint8", "int8", "uint16", "int16", "uint32", "int32" };

    for (int i = 0; i < 8; ++i)
        if (t == names[i]) {
            return i;
        }

    throw std::runtime_error("unsupported _ArrayType_ " + t);
}

void voxelsize_from_affine(Tpm& t) {
    for (int c = 0; c < 3; ++c) {
        const double n = std::sqrt(t.affine[c] * t.affine[c] + t.affine[4 + c] * t.affine[4 + c] +
                                   t.affine[8 + c] * t.affine[8 + c]);

        if (n > 0) {
            t.voxelsize[c] = n;
        }
    }
}

// A SAX DOM builder for BJData that stores the (huge) uint8 arrays of the
// _ArrayZipData_ / _ArrayData_ keys as one json binary instead of one json node
// per byte (json::from_bjdata took 6 GB for a 127 MB TPM).
class LeanDom {
  public:
    json root;

    bool null() {
        return put(json(nullptr));
    }
    bool boolean(bool v) {
        return put(json(v));
    }
    bool number_integer(json::number_integer_t v) {
        return cap_ ? byte(static_cast<int64_t>(v)) : put(json(v));
    }
    bool number_unsigned(json::number_unsigned_t v) {
        return cap_ ? byte(static_cast<int64_t>(v)) : put(json(v));
    }
    bool number_float(json::number_float_t v, const std::string&) {
        if (cap_) {   // not bytes after all: give up capturing
            spill();
        }

        return put(json(v));
    }
    bool string(std::string& v) {
        return put(json(std::move(v)));
    }
    bool binary(json::binary_t& v) {
        return put(json::binary(std::vector<std::uint8_t>(v.begin(), v.end())));
    }
    bool start_object(std::size_t) {
        return open(json::object());
    }
    bool key(std::string& k) {
        key_ = k;
        return true;
    }
    bool end_object() {
        return close();
    }
    bool start_array(std::size_t) {
        if (!cap_ && (key_ == "_ArrayZipData_" || key_ == "_ArrayData_") && !stack_.empty() &&
                stack_.back()->is_object()) {
            cap_ = true;
            depth_ = 0;
            bytes_.clear();
            ckey_ = key_;
            return true;
        }

        if (cap_) {   // nested: not a flat byte array
            spill();
        }

        return open(json::array());
    }
    bool end_array() {
        if (cap_ && depth_ == 0) {
            cap_ = false;
            key_ = ckey_;
            return put(json::binary(std::move(bytes_)));
        }

        return close();
    }
    bool parse_error(std::size_t pos, const std::string&, const nlohmann::detail::exception& e) {
        throw std::runtime_error("BJData parse error at byte " + std::to_string(pos) + ": " + e.what());
    }

  private:
    std::vector<json*> stack_;
    std::string key_, ckey_;
    bool cap_ = false;
    int depth_ = 0;
    std::vector<uint8_t> bytes_;

    bool byte(int64_t v) {
        if (v < 0 || v > 255) {
            spill();
            return put(json(v));
        }

        bytes_.push_back(static_cast<uint8_t>(v));
        return true;
    }
    // a captured array that is not bytes: replay it as an ordinary json array
    void spill() {
        cap_ = false;
        key_ = ckey_;
        json a = json::array();

        for (uint8_t b : bytes_) {
            a.push_back(b);
        }

        bytes_.clear();
        open(std::move(a));
    }
    bool put(json v) {
        if (stack_.empty()) {
            root = std::move(v);
            return true;
        }

        json& top = *stack_.back();

        if (top.is_object()) {
            top[key_] = std::move(v);
        } else {
            top.push_back(std::move(v));
        }

        return true;
    }
    bool open(json v) {
        if (stack_.empty()) {
            root = std::move(v);
            stack_.push_back(&root);
            return true;
        }

        json& top = *stack_.back();

        if (top.is_object()) {
            top[key_] = std::move(v);
            stack_.push_back(&top[key_]);
        } else {
            top.push_back(std::move(v));
            stack_.push_back(&top.back());
        }

        return true;
    }
    bool close() {
        stack_.pop_back();
        return true;
    }
};

Tpm load_jnifti(const std::string& path) {
    const std::vector<uint8_t> bytes = slurp(path);
    const bool bin = ends_with(lower(path), ".bnii");
    json root;

    if (bin) {
        LeanDom dom;
        json::sax_parse(bytes.begin(), bytes.end(), &dom, nlohmann::detail::input_format_t::bjdata);
        root = std::move(dom.root);
    } else {
        root = json::parse(bytes.begin(), bytes.end());
    }

    if (!root.contains("NIFTIHeader") || !root.contains("NIFTIData")) {
        throw std::runtime_error("not a JNIfTI file (no NIFTIHeader / NIFTIData): " + path);
    }

    const json& hdr = root["NIFTIHeader"];
    const json& nd = root["NIFTIData"];
    std::vector<double> dim = nd.is_object() && nd.contains("_ArraySize_") ? numbers(nd["_ArraySize_"])
                              : numbers(hdr.at("Dim"));

    if (dim.size() != 4 || dim[3] < 2) {
        throw std::invalid_argument("not a 4-D TPM");
    }

    Tpm t;
    t.nx = static_cast<int>(dim[0]);
    t.ny = static_cast<int>(dim[1]);
    t.nz = static_cast<int>(dim[2]);
    t.C = static_cast<int>(dim[3]);

    if (hdr.contains("VoxelSize")) {
        const std::vector<double> vs = numbers(hdr["VoxelSize"]);

        for (int i = 0; i < 3 && i < static_cast<int>(vs.size()); ++i) {
            t.voxelsize[i] = vs[i];
        }
    }

    if (hdr.contains("Affine")) {   // 3 x 4 or 4 x 4, row-major
        const std::vector<double> A = numbers(hdr["Affine"]);

        if (A.size() == 12 || A.size() == 16) {
            for (int k = 0; k < 12; ++k) {
                t.affine[k] = A[k];
            }

            voxelsize_from_affine(t);
        }
    } else {
        for (int c = 0; c < 3; ++c) {
            t.affine[5 * c] = t.voxelsize[c];
        }
    }

    // channel names: NIFTIHeader._DataInfo_.LabelTable {"0": {"Label": ...}, ...}
    t.names.assign(t.C, "");

    if (hdr.contains("_DataInfo_") && hdr["_DataInfo_"].contains("LabelTable")) {
        for (auto it = hdr["_DataInfo_"]["LabelTable"].begin(); it != hdr["_DataInfo_"]["LabelTable"].end(); ++it) {
            const int c = std::atoi(it.key().c_str());

            if (c >= 0 && c < t.C && it.value().is_object() && it.value().contains("Label")) {
                t.names[c] = it.value()["Label"].get<std::string>();
            }
        }
    }

    const int code = jdata_type(nd.value("_ArrayType_", std::string("single")));
    std::vector<uint8_t> raw = array_bytes(nd);
    const int use = nd.contains("_ArrayZipData_") || (nd.contains("_ArrayData_") && nd["_ArrayData_"].is_binary())
                    ? code : 1;
    const size_t nv = t.nv(), ne = nv * t.C;

    if (raw.size() != ne * kElemSize[use]) {
        throw std::runtime_error("NIFTIData: decoded size mismatch in " + path);
    }

    // [X][Y][Z][C] row-major (channel fastest) -> channel-major, x fastest
    const Typed src{ raw.data(), use };
    t.p.resize(ne);
    #pragma omp parallel for schedule(static)

    for (int x = 0; x < t.nx; ++x)
        for (int y = 0; y < t.ny; ++y)
            for (int z = 0; z < t.nz; ++z) {
                const size_t i = ((static_cast<size_t>(x) * t.ny + y) * t.nz + z) * t.C;
                const size_t o = (static_cast<size_t>(z) * t.ny + y) * t.nx + x;

                for (int c = 0; c < t.C; ++c) {
                    t.p[c * nv + o] = src.at(i + c);
                }
            }

    return t;
}

Tpm load_nifti(const std::string& path) {
    std::vector<uint8_t> b = slurp(path);

    if (b.size() >= 2 && b[0] == 0x1F && b[1] == 0x8B) {
        b = zdec(b.data(), b.size(), zmGzip);
    }

    if (b.size() < 352) {
        throw std::runtime_error("not a NIfTI-1 file: " + path);
    }

    int32_t sizeof_hdr;
    std::memcpy(&sizeof_hdr, b.data(), 4);

    if (sizeof_hdr != 348) {
        throw std::runtime_error("unsupported NIfTI (not little-endian NIfTI-1): " + path);
    }

    auto i16 = [&](size_t o) {
        int16_t v;
        std::memcpy(&v, b.data() + o, 2);
        return v;
    };
    auto f32 = [&](size_t o) {
        float v;
        std::memcpy(&v, b.data() + o, 4);
        return v;
    };

    if (i16(40) != 4 || i16(48) < 2) {
        throw std::invalid_argument("not a 4-D TPM");
    }

    Tpm t;
    t.nx = i16(42);
    t.ny = i16(44);
    t.nz = i16(46);
    t.C = i16(48);
    const int dt = i16(70);
    const int code = dt == 16 ? 0 : dt == 64 ? 1 : dt == 2 ? 2 : dt == 256 ? 3 : dt == 512 ? 4 : dt == 4 ? 5 :
                     dt == 768 ? 6 : dt == 8 ? 7 : -1;

    if (code < 0) {
        throw std::runtime_error("unsupported NIfTI datatype " + std::to_string(dt));
    }

    const size_t off = static_cast<size_t>(f32(108));
    float slope = f32(112);
    const float inter = f32(116);

    if (slope == 0.0f || !std::isfinite(slope)) {
        slope = 1.0f;
    }

    for (int c = 0; c < 3; ++c) {
        t.voxelsize[c] = std::fabs(f32(80 + 4 * c));
    }

    if (i16(254) > 0) {   // sform
        for (int k = 0; k < 12; ++k) {
            t.affine[k] = f32(280 + 4 * k);
        }

        voxelsize_from_affine(t);
    } else {   // (qform rotations are not decoded: axis-aligned pixdim)
        for (int c = 0; c < 3; ++c) {
            t.affine[5 * c] = t.voxelsize[c];
        }
    }

    const size_t ne = t.nv() * t.C;

    if (b.size() < off + ne * kElemSize[code]) {
        throw std::runtime_error("NIfTI: truncated data in " + path);
    }

    const Typed src{ b.data() + off, code };   // x fastest, then y, z, channel: as Tpm
    t.p.resize(ne);
    #pragma omp parallel for schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(ne); ++i) {
        t.p[i] = src.at(i) * slope + inter;
    }

    t.names.assign(t.C, "");
    return t;
}

bool exterior_name(const std::string& s) {
    const std::string n = lower(s);
    return n == "background" || n == "air" || n == "bg" || n == "outside" || n == "exterior" || n == "none";
}

void smooth3(std::vector<float>& f, size_t off, int nx, int ny, int nz, float sigma) {
    const int R = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> w(2 * R + 1);
    float ws = 0.0f;

    for (int d = -R; d <= R; ++d) {
        w[d + R] = std::exp(-0.5f * d * d / (sigma * sigma));
        ws += w[d + R];
    }

    for (float& x : w) {
        x /= ws;
    }

    const size_t nv = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> tmp(nv);
    const int64_t nxy = static_cast<int64_t>(nx) * ny;
    const int64_t st[3] = { 1, nx, nxy };
    const int len[3] = { nx, ny, nz };
    float* a = f.data() + off;

    for (int ax = 0; ax < 3; ++ax) {
        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            const int c = static_cast<int>((v / st[ax]) % len[ax]);
            float acc = 0.0f;

            for (int d = -R; d <= R; ++d) {
                const int cc = std::min(len[ax] - 1, std::max(0, c + d));
                acc += w[d + R] * a[v + (cc - c) * st[ax]];
            }

            tmp[v] = acc;
        }

        std::copy(tmp.begin(), tmp.end(), a);
    }
}

}  // namespace

Tpm load_tpm(const std::string& path) {
    const std::string p = lower(path);

    if (ends_with(p, ".jnii") || ends_with(p, ".bnii")) {
        return load_jnifti(path);
    }

    if (ends_with(p, ".nii") || ends_with(p, ".nii.gz")) {
        return load_nifti(path);
    }

    throw std::runtime_error("unsupported TPM extension (want .jnii/.bnii/.nii/.nii.gz): " + path);
}

bool is_tpm_file(const std::string& path) {
    const std::string p = lower(path);

    try {
        if (ends_with(p, ".jnii")) {   // the header comes first: scan its start
            std::ifstream f(path, std::ios::binary);
            std::string head(65536, '\0');
            f.read(&head[0], static_cast<std::streamsize>(head.size()));
            head.resize(static_cast<size_t>(f.gcount()));
            const size_t k = head.find("\"Dim\"");

            if (k == std::string::npos) {
                return false;
            }

            const size_t a = head.find('[', k), e = head.find(']', k);
            return a != std::string::npos && e != std::string::npos &&
                   std::count(head.begin() + a, head.begin() + e, ',') == 3;
        }

        if (ends_with(p, ".nii") || ends_with(p, ".nii.gz")) {
            std::vector<uint8_t> b = slurp(path);   // (a .nii.gz is inflated: header at the front)

            if (b.size() >= 2 && b[0] == 0x1F && b[1] == 0x8B) {
                b = zdec(b.data(), b.size(), zmGzip);
            }

            int16_t d0 = 0, d4 = 0;

            if (b.size() >= 50) {
                std::memcpy(&d0, b.data() + 40, 2);
                std::memcpy(&d4, b.data() + 48, 2);
            }

            return d0 == 4 && d4 > 1;
        }

        if (ends_with(p, ".bnii")) {
            LeanDom dom;
            const std::vector<uint8_t> bytes = slurp(path);
            json::sax_parse(bytes.begin(), bytes.end(), &dom, nlohmann::detail::input_format_t::bjdata);
            const json& root = dom.root;
            const json& nd = root.at("NIFTIData");
            const std::vector<double> dim = nd.is_object() && nd.contains("_ArraySize_") ? numbers(nd["_ArraySize_"])
                                            : numbers(root.at("NIFTIHeader").at("Dim"));
            return dim.size() == 4 && dim[3] > 1;
        }
    } catch (const std::exception&) {
        return false;
    }

    return false;
}

std::vector<int> apply_tpm(const Tpm& t, const TpmOptions& o, LabelVolume& lv, size_t* filled) {
    if (t.C < 1 || t.p.size() != t.nv() * t.C) {
        throw std::runtime_error("apply_tpm: empty or malformed TPM");
    }

    const size_t nv = t.nv();
    std::vector<int> map;

    if (o.spm6) {
        if (t.C != 18) {
            throw std::runtime_error("tpm spm6: needs the 18 siamize classes, got " + std::to_string(t.C));
        }

        for (int c = 0; c < 18; ++c) {   // SPM order GM WM CSF Bone Soft Air -> labels 1..5, Air = 0
            const int s = siam::SIAM18_TO_SPM6[c];
            map.push_back(s == siam::SPM6_AIR_CHANNEL ? 0 : s + 1);
        }
    } else if (!o.map.empty()) {
        if (static_cast<int>(o.map.size()) != t.C) {
            throw std::runtime_error("tpm map: " + std::to_string(o.map.size()) + " labels for " +
                                     std::to_string(t.C) + " channels");
        }

        map = o.map;
    } else {
        std::vector<char> ext(t.C, 0);

        if (!o.exterior.empty()) {
            for (int c : o.exterior) {
                if (c < 0 || c >= t.C) {
                    throw std::runtime_error("tpm exterior: channel " + std::to_string(c) + " out of range");
                }

                ext[c] = 1;
            }
        } else {
            for (int c = 0; c < t.C && c < static_cast<int>(t.names.size()); ++c) {
                ext[c] = exterior_name(t.names[c]);
            }
        }

        int next = 1;

        for (int c = 0; c < t.C; ++c) {
            map.push_back(ext[c] ? 0 : next++);
        }
    }

    int nlab = 1;

    for (int l : map) {
        if (l < 0 || l > 65534) {
            throw std::runtime_error("tpm map: bad label " + std::to_string(l));
        }

        nlab = std::max(nlab, l + 1);
    }

    const bool has_ext = std::find(map.begin(), map.end(), 0) != map.end();
    lv = LabelVolume();
    lv.nx = t.nx;
    lv.ny = t.ny;
    lv.nz = t.nz;
    lv.voxelsize = t.voxelsize;
    lv.affine = t.affine;
    lv.nprob = nlab;
    lv.prob.assign(static_cast<size_t>(nlab) * nv, 0.0f);

    for (int c = 0; c < t.C; ++c) {
        float* dst = lv.prob.data() + static_cast<size_t>(map[c]) * nv;
        const float* src = t.p.data() + static_cast<size_t>(c) * nv;
        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            dst[v] += std::max(0.0f, src[v]);
        }
    }

    if (o.sigma > 0.0f) {
        for (int l = 0; l < nlab; ++l) {
            smooth3(lv.prob, static_cast<size_t>(l) * nv, t.nx, t.ny, t.nz, o.sigma);
        }
    }

    if (!has_ext) {   // no exterior channel: whatever the tissues leave
        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            float s = 0.0f;

            for (int l = 1; l < nlab; ++l) {
                s += lv.prob[static_cast<size_t>(l) * nv + v];
            }

            lv.prob[v] = std::min(1.0f, std::max(0.0f, 1.0f - s));
        }
    }

    lv.data.assign(nv, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        int best = 0;
        float bp = lv.prob[v];

        for (int l = 1; l < nlab; ++l) {
            const float q = lv.prob[static_cast<size_t>(l) * nv + v];

            if (q > bp) {
                bp = q;
                best = l;
            }
        }

        lv.data[v] = static_cast<uint16_t>(best);
    }

    size_t nfill = 0;

    if (o.fill_holes) {
        // exterior reachable from the volume boundary (6-connected), the rest is a hole
        const int nx = t.nx, ny = t.ny, nz = t.nz;
        std::vector<uint8_t> st(nv, 0);   // 1 = outside (reached), 2 = queued hole fill
        std::vector<uint32_t> q;
        auto at = [&](int x, int y, int z) {
            return static_cast<size_t>(x) + static_cast<size_t>(nx) * (y + static_cast<size_t>(ny) * z);
        };

        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    if (x && y && z && x < nx - 1 && y < ny - 1 && z < nz - 1) {
                        continue;
                    }

                    const size_t v = at(x, y, z);

                    if (lv.data[v] == 0 && !st[v]) {
                        st[v] = 1;
                        q.push_back(static_cast<uint32_t>(v));
                    }
                }

        const int64_t nxy = static_cast<int64_t>(nx) * ny;
        const int64_t step[3] = { 1, nx, nxy };
        const int dim[3] = { nx, ny, nz };

        for (size_t h = 0; h < q.size(); ++h) {
            const size_t v = q[h];
            const int c[3] = { static_cast<int>(v % nx), static_cast<int>((v / nx) % ny), static_cast<int>(v / (static_cast<size_t>(nx) * ny)) };

            for (int a = 0; a < 3; ++a)
                for (int sg = -1; sg <= 1; sg += 2) {
                    const int cc = c[a] + sg;

                    if (cc < 0 || cc >= dim[a]) {
                        continue;
                    }

                    const size_t u = v + sg * step[a];

                    if (lv.data[u] == 0 && !st[u]) {
                        st[u] = 1;
                        q.push_back(static_cast<uint32_t>(u));
                    }
                }
        }

        // holes: multi-source BFS from the surrounding tissue (nearest label wins)
        q.clear();

        for (size_t v = 0; v < nv; ++v) {
            if (lv.data[v] != 0) {
                q.push_back(static_cast<uint32_t>(v));
            }
        }

        for (size_t v = 0; v < nv; ++v) {
            nfill += lv.data[v] == 0 && !st[v];
        }

        if (nfill) {
            for (size_t h = 0; h < q.size(); ++h) {
                const size_t v = q[h];
                const int c[3] = { static_cast<int>(v % nx), static_cast<int>((v / nx) % ny), static_cast<int>(v / (static_cast<size_t>(nx) * ny)) };

                for (int a = 0; a < 3; ++a)
                    for (int sg = -1; sg <= 1; sg += 2) {
                        const int cc = c[a] + sg;

                        if (cc < 0 || cc >= dim[a]) {
                            continue;
                        }

                        const size_t u = v + sg * step[a];

                        if (lv.data[u] == 0 && !st[u]) {   // an unfilled hole voxel
                            st[u] = 2;
                            lv.data[u] = lv.data[v];

                            for (int l = 0; l < nlab; ++l) {
                                lv.prob[static_cast<size_t>(l) * nv + u] = l == lv.data[u] ? 1.0f : 0.0f;
                            }

                            q.push_back(static_cast<uint32_t>(u));
                        }
                    }
            }
        }
    }

    if (o.fill_holes) {
        // deep inside the tissue (> 2 voxels from any exterior voxel) no exterior
        // can exist: an atlas' tissue probabilities that do not sum to 1 there (or a
        // network's residual background) would otherwise leave near-exterior spots
        // inside the head. Zero p_0 there and renormalize the tissues; the band at
        // the outer surface keeps its values, so the surface does not move.
        const int nx = t.nx, ny = t.ny, nz = t.nz;
        std::vector<uint8_t> dist(nv, 255);
        std::vector<uint32_t> q;

        for (size_t v = 0; v < nv; ++v) {
            if (lv.data[v] == 0) {
                dist[v] = 0;
                q.push_back(static_cast<uint32_t>(v));
            }
        }

        const int64_t nxy = static_cast<int64_t>(nx) * ny;
        const int64_t step[3] = { 1, nx, nxy };
        const int dim[3] = { nx, ny, nz };

        for (size_t h = 0; h < q.size(); ++h) {
            const size_t v = q[h];

            if (dist[v] >= 3) {
                continue;
            }

            const int c[3] = { static_cast<int>(v % nx), static_cast<int>((v / nx) % ny), static_cast<int>(v / (static_cast<size_t>(nx) * ny)) };

            for (int a = 0; a < 3; ++a)
                for (int sg = -1; sg <= 1; sg += 2) {
                    const int cc = c[a] + sg;

                    if (cc < 0 || cc >= dim[a]) {
                        continue;
                    }

                    const size_t u = v + sg * step[a];

                    if (dist[u] == 255) {
                        dist[u] = static_cast<uint8_t>(dist[v] + 1);
                        q.push_back(static_cast<uint32_t>(u));
                    }
                }
        }

        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            if (dist[v] <= 2 || lv.prob[v] <= 0.0f) {
                continue;
            }

            float s = 0.0f;

            for (int l = 1; l < nlab; ++l) {
                s += lv.prob[static_cast<size_t>(l) * nv + v];
            }

            lv.prob[v] = 0.0f;

            if (s > 0.0f) {
                for (int l = 1; l < nlab; ++l) {
                    lv.prob[static_cast<size_t>(l) * nv + v] /= s;
                }
            }
        }
    }

    if (filled) {
        *filled = nfill;
    }

    const double vv = lv.voxelsize[0] * lv.voxelsize[1] * lv.voxelsize[2];
    lv.soft_volume.assign(nlab, 0.0);

    for (int l = 0; l < nlab; ++l) {
        double s = 0.0;
        const float* P = lv.prob.data() + static_cast<size_t>(l) * nv;
        #pragma omp parallel for reduction(+ : s) schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            s += P[v];
        }

        lv.soft_volume[l] = s * vv;
    }

    if (!o.fields) {   // the labels only: meshed like a label volume
        std::vector<float>().swap(lv.prob);
        lv.nprob = 0;
    }

    int m = 0;

    for (uint16_t l : lv.data) {
        m = std::max<int>(m, l);
    }

    lv.maxlabel = m;
    return map;
}

}  // namespace tn
