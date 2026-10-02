// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_tpm.cpp -- see v2m_tpm.h. The JNIfTI part follows gpu_brain2mesh's b2m_tpm
// (JData annotated arrays, zlib / base64 through zlibmt.h); the NIfTI-1 part reads a
// 4-D .nii[.gz] directly (the siamize reader is 3-D only). The volume stays in
// its file's voxel order; the affine maps it to world coordinates.

#include "v2m_tpm.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "siam.h"   // SIAM18_TO_SPM6
#include "zlibmt.h"   // zlib / gzip (multithreaded) + base64: mimamo

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
        b = zlibmt::base64_decode(s.data(), s.size());
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
// (pay: the payload bytes in place, found by find_bj_payload -- then the key's
// value is the emptied array; nexp > 0: the decoded size expected)
std::vector<uint8_t> array_bytes(const json& a, const uint8_t* pay = nullptr, size_t npay = 0, size_t nexp = 0) {
    if (a.contains("_ArrayZipData_")) {
        const std::string zt = a.value("_ArrayZipType_", std::string("zlib"));

        if (zt != "zlib" && zt != "gzip") {
            throw std::runtime_error("unsupported _ArrayZipType_ " + zt);
        }

        std::vector<uint8_t> zb;

        if (!pay) {
            zb = payload_bytes(a["_ArrayZipData_"]);
            pay = zb.data();
            npay = zb.size();
        }

        return zt == "gzip" ? zlibmt::gzip_decompress(pay, npay, nexp) : zlibmt::zlib_decompress(pay, npay, nexp);
    }

    if (pay && a.contains("_ArrayData_")) {
        return std::vector<uint8_t>(pay, pay + npay);
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

// The voxel payload of a BJData JNIfTI, located in the raw bytes: the value of
// the first _ArrayZipData_ / _ArrayData_ key that is a strongly-typed uint8 array
// ([$U#<count> then the bytes). The SAX parse would otherwise take one callback
// per payload byte (5.5 s for a 1.1 GB TPM); instead the small JSON around it is
// parsed with the array emptied, and the bytes are used in place.
struct BjPayload {
    size_t at = 0;    // the '[' of the array
    size_t off = 0;   // its first byte
    size_t len = 0;   // its byte count
};

bool find_bj_payload(const uint8_t* b, size_t n, BjPayload& p) {
    static const char* keys[2] = { "_ArrayZipData_", "_ArrayData_" };

    for (const char* key : keys) {
        const size_t kl = std::strlen(key);

        for (size_t k = 2; k + kl + 6 <= n; ++k) {
            if (b[k] != '_' || std::memcmp(b + k, key, kl) != 0 || b[k - 1] != kl || (b[k - 2] != 'i' && b[k - 2] != 'U')) {
                continue;
            }

            size_t q = k + kl;

            if (b[q] != '[' || b[q + 1] != '$' || b[q + 2] != 'U' || b[q + 3] != '#') {
                continue;
            }

            const uint8_t m = b[q + 4];
            const int w = m == 'i' || m == 'U' ? 1 : m == 'I' || m == 'u' ? 2 : m == 'l' || m == 'm' ? 4 : m == 'L' || m == 'M' ? 8 : 0;

            if (w == 0 || q + 5 + w > n) {
                continue;
            }

            uint64_t c = 0;
            std::memcpy(&c, b + q + 5, static_cast<size_t>(w));   // (little-endian, as BJData)

            if ((m == 'i' && (c & 0x80)) || (m == 'I' && (c & 0x8000)) || (m == 'l' && (c & 0x80000000u))) {
                continue;   // (a negative count)
            }

            p.at = q;
            p.off = q + 5 + static_cast<size_t>(w);
            p.len = static_cast<size_t>(c);
            return true;
        }
    }

    return false;
}

// parse the BJData around the payload (b[0, p.at) + "[]" + tail): its array empty
json parse_bj_around(const uint8_t* b, const BjPayload& p, const uint8_t* tail, size_t ntail) {
    std::vector<uint8_t> small(b, b + p.at);
    small.push_back('[');
    small.push_back(']');
    small.insert(small.end(), tail, tail + ntail);
    LeanDom dom;
    json::sax_parse(small.begin(), small.end(), &dom, nlohmann::detail::input_format_t::bjdata);
    return std::move(dom.root);
}

Tpm load_jnifti(const std::string& path) {
    const std::vector<uint8_t> bytes = slurp(path);
    const bool bin = ends_with(lower(path), ".bnii");
    json root;

    BjPayload pay;
    const bool inplace = bin && find_bj_payload(bytes.data(), bytes.size(), pay) && pay.off + pay.len <= bytes.size();

    if (inplace) {
        root = parse_bj_around(bytes.data(), pay, bytes.data() + pay.off + pay.len, bytes.size() - pay.off - pay.len);
    } else if (bin) {
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
    const size_t nexp = t.nv() * t.C * kElemSize[code];
    std::vector<uint8_t> raw = inplace ? array_bytes(nd, bytes.data() + pay.off, pay.len, nexp) : array_bytes(nd);
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
        b = zlibmt::gzip_decompress(b.data(), b.size());
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

static void set_label_thresh(TpmOptions& o, int l, double t) {
    if (l < 0 || l > 65534 || !(t > 0.0 && t < 1.0)) {
        throw std::runtime_error("tpm thresh: want label >= 0 and 0 < threshold < 1");
    }

    if (static_cast<int>(o.thresh.size()) <= l) {
        o.thresh.resize(l + 1, 0.0f);
    }

    o.thresh[l] = static_cast<float>(t);
}

void parse_tpm_thresh(const std::string& s, TpmOptions& o) {
    size_t p0 = 0;

    while (p0 <= s.size()) {
        const size_t p1 = s.find(',', p0);
        const std::string item = s.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
        const size_t c = item.find(':');
        char* end = nullptr;

        if (c == std::string::npos) {
            const double t = std::strtod(item.c_str(), &end);

            if (item.empty() || *end != '\0' || !(t > 0.0 && t < 1.0)) {
                throw std::runtime_error("tpm thresh: bad item '" + item + "' (want T or L:T, 0 < T < 1)");
            }

            o.thresh_all = static_cast<float>(t);
        } else {
            const long l = std::strtol(item.substr(0, c).c_str(), &end, 10);
            const bool lok = c > 0 && *end == '\0';
            const double t = std::strtod(item.substr(c + 1).c_str(), &end);

            if (!lok || c + 1 >= item.size() || *end != '\0') {
                throw std::runtime_error("tpm thresh: bad item '" + item + "' (want T or L:T, 0 < T < 1)");
            }

            set_label_thresh(o, static_cast<int>(l), t);
        }

        if (p1 == std::string::npos) {
            break;
        }

        p0 = p1 + 1;
    }
}

void parse_tpm_pair(const std::string& s, TpmOptions& o) {
    size_t p0 = 0;

    while (p0 <= s.size()) {
        const size_t p1 = std::min(s.find(',', p0), s.size());
        const std::string item = s.substr(p0, p1 - p0);
        p0 = p1 + 1;

        if (item.empty()) {
            continue;
        }

        int a = -1, b = -1;
        double t = 0;
        char tail = 0;

        if (std::sscanf(item.c_str(), "%d:%d:%lf%c", &a, &b, &t, &tail) != 3 || a < 0 || b < 0 || a == b || !(t > 0 && t < 1)) {
            throw std::runtime_error("tpm pair: bad item '" + item + "' (want A:B:T, labels A != B, 0 < T < 1)");
        }

        o.pair.push_back({ { static_cast<float>(a), static_cast<float>(b), static_cast<float>(t) } });
    }
}

void parse_tpm_gap(const std::string& s, TpmOptions& o) {
    size_t p0 = 0;

    while (p0 <= s.size()) {
        const size_t p1 = std::min(s.find(',', p0), s.size());
        const std::string item = s.substr(p0, p1 - p0);
        p0 = p1 + 1;

        if (item.empty()) {
            continue;
        }

        const std::string bad = "tpm gap: bad item '" + item + "' (want A:B:C[+C..]:D, labels A, B, C distinct, D > 0 mm)";
        TpmOptions::Gap g;
        int n = 0;
        char tail = 0;

        if (std::sscanf(item.c_str(), "%d:%d:%n", &g.a, &g.b, &n) != 2 || n == 0) {
            throw std::runtime_error(bad);
        }

        const size_t q = item.rfind(':');

        if (q == std::string::npos || q < static_cast<size_t>(n) ||
            std::sscanf(item.c_str() + q + 1, "%f%c", &g.d, &tail) != 1 || !(g.d > 0.0f)) {
            throw std::runtime_error(bad);
        }

        const std::string cs = item.substr(n, q - n);
        size_t c0 = 0;

        while (c0 <= cs.size()) {
            const size_t c1 = std::min(cs.find('+', c0), cs.size());
            int c = -1;

            if (std::sscanf(cs.substr(c0, c1 - c0).c_str(), "%d%c", &c, &tail) != 1 || c < 0 || c == g.a || c == g.b) {
                throw std::runtime_error(bad);
            }

            g.c.push_back(c);
            c0 = c1 + 1;
        }

        if (g.a < 0 || g.b < 0 || g.a == g.b) {
            throw std::runtime_error(bad);
        }

        o.gap.push_back(g);
    }
}

void set_tpm_thresh(const std::vector<double>& v, TpmOptions& o) {
    if (v.size() == 1) {
        if (!(v[0] > 0.0 && v[0] < 1.0)) {
            throw std::runtime_error("tpm thresh: want 0 < threshold < 1");
        }

        o.thresh_all = static_cast<float>(v[0]);
        return;
    }

    if (v.size() % 2) {
        throw std::runtime_error("tpm thresh: want one threshold or (label, threshold) pairs");
    }

    for (size_t k = 0; k + 1 < v.size(); k += 2) {
        set_label_thresh(o, static_cast<int>(std::lround(v[k])), v[k + 1]);
    }
}

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
                b = zlibmt::gzip_decompress(b.data(), b.size());
            }

            int16_t d0 = 0, d4 = 0;

            if (b.size() >= 50) {
                std::memcpy(&d0, b.data() + 40, 2);
                std::memcpy(&d4, b.data() + 48, 2);
            }

            return d0 == 4 && d4 > 1;
        }

        if (ends_with(p, ".bnii")) {
            // the header and whatever follows the payload, not the payload itself
            // (a full parse took 5.5 s for a 1.1 GB TPM)
            json root;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            const size_t fsz = f ? static_cast<size_t>(f.tellg()) : 0;
            std::vector<uint8_t> head(std::min<size_t>(fsz, size_t(1) << 20));
            f.seekg(0);
            BjPayload pay;

            if (f && f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size())) &&
                    find_bj_payload(head.data(), head.size(), pay) && pay.off + pay.len <= fsz) {
                std::vector<uint8_t> tail(fsz - pay.off - pay.len);
                f.seekg(static_cast<std::streamoff>(pay.off + pay.len));

                if (!tail.empty() && !f.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail.size()))) {
                    return false;
                }

                root = parse_bj_around(head.data(), pay, tail.data(), tail.size());
            } else {
                LeanDom dom;
                const std::vector<uint8_t> bytes = slurp(path);
                json::sax_parse(bytes.begin(), bytes.end(), &dom, nlohmann::detail::input_format_t::bjdata);
                root = std::move(dom.root);
            }

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

    // V2M_TPM_TIMING: the time of each phase
    static const bool ptiming = std::getenv("V2M_TPM_TIMING") != nullptr;
    auto tph = std::chrono::steady_clock::now();
    auto phase = [&](const char* what) {
        if (ptiming) {
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[tpmt] %-12s %8.0f ms\n", what, std::chrono::duration<double, std::milli>(now - tph).count());
            tph = now;
        }
    };
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
    // integer maps (e.g. 0..255): probabilities are 0..1 here (the thresholds, the
    // exterior as 1 - sum and the renormalizations assume it)
    float vmax = 0.0f;
    #pragma omp parallel for reduction(max : vmax) schedule(static)

    for (int64_t i = 0; i < static_cast<int64_t>(t.p.size()); ++i) {
        vmax = std::max(vmax, t.p[i]);
    }

    const float vscale = vmax > 1.5f ? 1.0f / (vmax <= 255.0f ? 255.0f : vmax) : 1.0f;
    phase("vmax");
    // the threshold bias b_l = 0.5 - t_l (0: the plain argmax)
    std::vector<float> bias(nlab, 0.0f);
    bool biased = false;

    for (int l = 0; l < nlab; ++l) {
        float tl = l > 0 && o.thresh_all > 0.0f ? o.thresh_all : 0.0f;

        if (l < static_cast<int>(o.thresh.size()) && o.thresh[l] > 0.0f) {
            tl = o.thresh[l];
        }

        if (tl > 0.0f) {
            if (tl >= 1.0f) {
                throw std::runtime_error("tpm thresh: label " + std::to_string(l) + " threshold must be in (0, 1)");
            }

            bias[l] = 0.5f - tl;
            biased = biased || bias[l] != 0.0f;
        }
    }

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
            dst[v] += std::max(0.0f, src[v] * vscale);
        }
    }

    phase("merge");
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

    phase("sigma/ext");
    // the pair thresholds: label A's per-voxel bias, (0.5 - T) x B's share of A's
    // competition (from the probabilities as they are, before any shift)
    std::vector<std::vector<float>> pbias(static_cast<size_t>(nlab));

    for (const auto& pr : o.pair) {
        const int a = static_cast<int>(pr[0]), b = static_cast<int>(pr[1]);

        if (a >= nlab || b >= nlab) {
            throw std::runtime_error("tpm pair: label " + std::to_string(std::max(a, b)) + " is not in the map (labels 0.." +
                                     std::to_string(nlab - 1) + ")");
        }

        std::vector<float>& pa = pbias[static_cast<size_t>(a)];
        pa.resize(nv, 0.0f);
        const float k = 0.5f - pr[2];
        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            float rest = 0.0f;

            for (int l = 0; l < nlab; ++l)
                if (l != a) {
                    rest += lv.prob[static_cast<size_t>(l) * nv + v];
                }

            if (rest > 1e-6f) {
                pa[static_cast<size_t>(v)] += k * lv.prob[static_cast<size_t>(b) * nv + v] / rest;
            }
        }

        biased = true;
    }

    auto pb = [&](int l, int64_t v) {
        return pbias[static_cast<size_t>(l)].empty() ? 0.0f : pbias[static_cast<size_t>(l)][static_cast<size_t>(v)];
    };
    lv.data.assign(nv, 0);
    #pragma omp parallel for schedule(static)

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        int best = 0;
        float bp = lv.prob[v] + bias[0] + pb(0, v);

        for (int l = 1; l < nlab; ++l) {
            const float q = lv.prob[static_cast<size_t>(l) * nv + v] + bias[l] + pb(l, v);

            if (q > bp) {
                bp = q;
                best = l;
            }
        }

        lv.data[v] = static_cast<uint16_t>(best);
    }

    phase("argmax");
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

    phase("fill-holes");
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

    phase("deep-ext");
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

    phase("soft-vol");
    // the minimum gaps: the A voxels whose centre is nearer than D + h to a C voxel's
    // (h the smallest voxel size: the interfaces lie half a voxel off the centres on
    // each side, so the B layer left between them is at least D) become B; their
    // probability moves to B, so the fields (--tpm-fields) agree (after the soft
    // volumes: they stay the map's, and the volume check shows what the gap took)
    for (const auto& g : o.gap) {
        const int nx = t.nx, ny = t.ny, nz = t.nz;
        std::vector<char> isc(static_cast<size_t>(nlab), 0);

        for (int c : g.c) {
            if (c >= nlab) {
                throw std::runtime_error("tpm gap: label " + std::to_string(c) + " is not in the map (labels 0.." +
                                         std::to_string(nlab - 1) + ")");
            }

            isc[static_cast<size_t>(c)] = 1;
        }

        if (g.a >= nlab || g.b >= nlab) {
            throw std::runtime_error("tpm gap: label " + std::to_string(std::max(g.a, g.b)) + " is not in the map (labels 0.." +
                                     std::to_string(nlab - 1) + ")");
        }

        const double h = std::min(lv.voxelsize[0], std::min(lv.voxelsize[1], lv.voxelsize[2]));
        const double reach = g.d + h;
        int r[3];

        for (int a = 0; a < 3; ++a) {
            r[a] = static_cast<int>(std::floor(reach / lv.voxelsize[a]));
        }

        struct Off {
            int dx, dy, dz;
            double d2;
        };
        std::vector<Off> ball;   // nearest first: most hits end the scan early

        for (int dz = -r[2]; dz <= r[2]; ++dz)
            for (int dy = -r[1]; dy <= r[1]; ++dy)
                for (int dx = -r[0]; dx <= r[0]; ++dx) {
                    const double ex = dx * lv.voxelsize[0], ey = dy * lv.voxelsize[1], ez = dz * lv.voxelsize[2];
                    const double d2 = ex * ex + ey * ey + ez * ez;

                    if ((dx || dy || dz) && d2 < reach * reach) {
                        ball.push_back({ dx, dy, dz, d2 });
                    }
                }

        std::sort(ball.begin(), ball.end(), [](const Off& p, const Off& q) { return p.d2 < q.d2; });
        std::vector<char> hit(nv, 0);
        size_t ncarve = 0;
        #pragma omp parallel for reduction(+ : ncarve) schedule(dynamic, 4)

        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const size_t v = static_cast<size_t>(x) + static_cast<size_t>(nx) * (y + static_cast<size_t>(ny) * z);

                    if (lv.data[v] != g.a) {
                        continue;
                    }

                    for (const Off& f : ball) {
                        const int X = x + f.dx, Y = y + f.dy, Z = z + f.dz;

                        if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) {
                            continue;
                        }

                        if (isc[lv.data[static_cast<size_t>(X) + static_cast<size_t>(nx) * (Y + static_cast<size_t>(ny) * Z)]]) {
                            hit[v] = 1;
                            ++ncarve;
                            break;
                        }
                    }
                }

        #pragma omp parallel for schedule(static)

        for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
            if (hit[v]) {
                float* pa = &lv.prob[static_cast<size_t>(g.a) * nv + v];
                lv.prob[static_cast<size_t>(g.b) * nv + v] += *pa;
                *pa = 0.0f;
                lv.data[v] = static_cast<uint16_t>(g.b);
            }
        }

        std::fprintf(stderr, "[tpm] gap %d:%d: %zu voxels of label %d -> %d (%.3g mm)\n", g.a, g.b, ncarve, g.a, g.b, g.d);
    }

    phase("gap");
    if (!o.fields) {   // the labels only: meshed like a label volume
        std::vector<float>().swap(lv.prob);
        lv.nprob = 0;
    } else if (biased) {   // the fields shifted like the argmax: interfaces at p_a - t_a = p_b - t_b
        for (int l = 0; l < nlab; ++l) {
            float* P = lv.prob.data() + static_cast<size_t>(l) * nv;
            const float b = bias[l];
            const float* Q = pbias[static_cast<size_t>(l)].empty() ? nullptr : pbias[static_cast<size_t>(l)].data();
            #pragma omp parallel for schedule(static)

            for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
                P[v] += b + (Q ? Q[v] : 0.0f);
            }
        }
    }

    int m = 0;

    for (uint16_t l : lv.data) {
        m = std::max<int>(m, l);
    }

    lv.maxlabel = m;
    return map;
}

}  // namespace tn
