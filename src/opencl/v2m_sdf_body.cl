// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_sdf_body.cl -- analytic signed distance functions of shape constructs
// (JSON shapes, v2m_sdfshape.h): one postfix program per label, evaluated with
// a fixed stack. Shared by the host (V2M_G empty) and the OpenCL device.
//
// Program layout (floats):
//   [0] N labels   [1] w (the field width: phi = clamp(0.5 + s / w))
//   [2..4] origin (world = grid mm + origin)   [5 .. 5+N-1] code offsets
//   [5+N] the crease blend radius of min / max (0: exact)   [6+N] the gap-closing radius (0: off)
//   [7+N] the cull margin (V2M_SDF_BBOX; its constants are -+ it)
//   [8+N] the brick table (0: none): nbx nby nbz voxel, then per brick (x
//         fastest) the offset of its N label code offsets -- its own program,
//         with only the objects near it
//   code: opcode, operands ...; V2M_SDF_END ends a label's code
// Values are s > 0 inside, world units; each primitive's gradient comes along
// (analytic, or central differences of that primitive alone), carried through
// max / min by the winning branch.

#define V2M_SDF_END 0
#define V2M_SDF_PRIM 1     // type, parameters
#define V2M_SDF_MAX 2      // union
#define V2M_SDF_MIN 3      // intersection
#define V2M_SDF_NEG 4      // complement
#define V2M_SDF_CONST 5    // value
#define V2M_SDF_BBOX 6     // lo[3] hi[3] len f: outside the box (+ margin m) -> push f m, skip len
#define V2M_SDF_MCONST 7   // f: push f x the cull margin (an object culled from a brick's program)
#define V2M_SDF_ADD 8      // a + b (--overlap split: s_a - s_b, with NEG)
#define V2M_SDF_SCALE 9    // f: x f

#define V2M_SDF_SPHERE 1     // c[3] r
#define V2M_SDF_BOX 2        // lo[3] hi[3] (axis-aligned)
#define V2M_SDF_CYL 3        // a[3] b[3] r (capped)
#define V2M_SDF_PLANE 4      // o[3] n[3] (unit): the half-space behind n
#define V2M_SDF_SLAB 5       // o[3] n[3] lo hi: lo < (p - o).n < hi
#define V2M_SDF_TORUS 6      // c[3] n[3] R r
#define V2M_SDF_ELLIPSOID 7  // c[3] r[3] rot[9] (row-major, world -> local)
#define V2M_SDF_CONE 8       // a[3] b[3] ra rb (capped frustum)
#define V2M_SDF_INFCYL 9     // o[3] n[3] r (infinite)

#define V2M_SDF_STACK 12   // (the host checks each program's depth: v2m_sdfshape.cpp)

inline int v2m_sdf_nparam(int type) {
    switch (type) {
    case V2M_SDF_SPHERE:
        return 4;

    case V2M_SDF_BOX:
        return 6;

    case V2M_SDF_CYL:
    case V2M_SDF_INFCYL:
        return 7;

    case V2M_SDF_PLANE:
        return 6;

    case V2M_SDF_SLAB:
    case V2M_SDF_TORUS:
    case V2M_SDF_CONE:
        return 8;

    case V2M_SDF_ELLIPSOID:
        return 15;

    default:
        return 0;
    }
}

// one primitive's value at (x, y, z) (world), s > 0 inside
inline float v2m_sdf_prim(int type, V2M_G const float* q, float x, float y, float z) {
    if (type == V2M_SDF_SPHERE) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        return q[3] - sqrt(dx * dx + dy * dy + dz * dz);
    }

    if (type == V2M_SDF_BOX) {   // exact box distance
        const float cx = 0.5f * (q[0] + q[3]), cy = 0.5f * (q[1] + q[4]), cz = 0.5f * (q[2] + q[5]);
        const float ux = fabs(x - cx) - 0.5f * fabs(q[3] - q[0]), uy = fabs(y - cy) - 0.5f * fabs(q[4] - q[1]),
                    uz = fabs(z - cz) - 0.5f * fabs(q[5] - q[2]);
        const float ox = fmax(ux, 0.0f), oy = fmax(uy, 0.0f), oz = fmax(uz, 0.0f);
        const float out = sqrt(ox * ox + oy * oy + oz * oz), in = fmin(fmax(ux, fmax(uy, uz)), 0.0f);
        return -(out + in);
    }

    if (type == V2M_SDF_PLANE) {
        return -((x - q[0]) * q[3] + (y - q[1]) * q[4] + (z - q[2]) * q[5]);
    }

    if (type == V2M_SDF_SLAB) {
        const float t = (x - q[0]) * q[3] + (y - q[1]) * q[4] + (z - q[2]) * q[5];
        return fmin(t - q[6], q[7] - t);
    }

    if (type == V2M_SDF_INFCYL) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        const float t = dx * q[3] + dy * q[4] + dz * q[5];
        const float rx = dx - t * q[3], ry = dy - t * q[4], rz = dz - t * q[5];
        return q[6] - sqrt(rx * rx + ry * ry + rz * rz);
    }

    if (type == V2M_SDF_TORUS) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        const float h = dx * q[3] + dy * q[4] + dz * q[5];
        const float rx = dx - h * q[3], ry = dy - h * q[4], rz = dz - h * q[5];
        const float a = sqrt(rx * rx + ry * ry + rz * rz) - q[6];
        return q[7] - sqrt(a * a + h * h);
    }

    if (type == V2M_SDF_CYL) {   // capped cylinder (exact; I. Quilez's formulation)
        const float bx = q[3] - q[0], by = q[4] - q[1], bz = q[5] - q[2];
        const float px = x - q[0], py = y - q[1], pz = z - q[2];
        const float baba = bx * bx + by * by + bz * bz, paba = px * bx + py * by + pz * bz;
        const float ex = px * baba - bx * paba, ey = py * baba - by * paba, ez = pz * baba - bz * paba;
        const float u = sqrt(ex * ex + ey * ey + ez * ez) - q[6] * baba;
        const float v = fabs(paba - baba * 0.5f) - baba * 0.5f;
        const float u2 = u * u, v2 = v * v * baba;
        const float dd = fmax(u, v) < 0.0f ? -fmin(u2, v2) : ((u > 0.0f ? u2 : 0.0f) + (v > 0.0f ? v2 : 0.0f));
        return -(dd < 0.0f ? -1.0f : 1.0f) * sqrt(fabs(dd)) / baba;
    }

    if (type == V2M_SDF_CONE) {   // capped frustum a (radius ra) .. b (radius rb) (exact; I. Quilez)
        const float bx = q[3] - q[0], by = q[4] - q[1], bz = q[5] - q[2];
        const float px = x - q[0], py = y - q[1], pz = z - q[2];
        const float ra = q[6], rb = q[7];
        const float rba = rb - ra, baba = bx * bx + by * by + bz * bz;
        const float papa = px * px + py * py + pz * pz, paba = (px * bx + py * by + pz * bz) / baba;
        const float xr = sqrt(fmax(papa - paba * paba * baba, 0.0f));
        const float cax = fmax(0.0f, xr - ((paba < 0.5f) ? ra : rb)), cay = fabs(paba - 0.5f) - 0.5f;
        const float k = rba * rba + baba;
        const float f = fmin(fmax((rba * (xr - ra) + paba * baba) / k, 0.0f), 1.0f);
        const float cbx = xr - ra - f * rba, cby = paba - f;
        const float sg = (cbx < 0.0f && cay < 0.0f) ? -1.0f : 1.0f;
        const float d2 = fmin(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba);
        return -sg * sqrt(d2);
    }

    if (type == V2M_SDF_ELLIPSOID) {   // a bound (exact on the surface): k0 (k0 - 1) / k1
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        const float lx = q[6] * dx + q[7] * dy + q[8] * dz, ly = q[9] * dx + q[10] * dy + q[11] * dz,
                    lz = q[12] * dx + q[13] * dy + q[14] * dz;
        const float ax = lx / q[3], ay = ly / q[4], az = lz / q[5];
        const float bx = ax / q[3], by = ay / q[4], bz = az / q[5];
        const float k0 = sqrt(ax * ax + ay * ay + az * az), k1 = sqrt(bx * bx + by * by + bz * bz);

        if (k1 < 1e-20f) {
            return fmin(q[3], fmin(q[4], q[5]));
        }

        return -k0 * (k0 - 1.0f) / k1;
    }

    return -1e30f;
}

// (host, debug: evaluations counted, [0] all, [1] with a gradient -- V2M_SDF_COUNT)
#ifndef __OPENCL_VERSION__
inline unsigned long long* v2m_sdf_calls() {
    static unsigned long long c[2] = { 0, 0 };
    return c;
}
inline bool v2m_sdf_counting() {   // (only when asked: the shared counter would stall the threads)
    static const bool on = std::getenv("V2M_SDF_COUNT") != nullptr;
    return on;
}
    #define V2M_SDF_COUNT(g) (v2m_sdf_counting() ? (++v2m_sdf_calls()[0], (g) ? ++v2m_sdf_calls()[1] : 0) : 0)
#else
    #define V2M_SDF_COUNT(g)
#endif

// (on the device: not inlined -- it is called from a dozen field queries, and
// inlined everywhere it made the program build take a minute)
#ifdef __OPENCL_VERSION__
    #define V2M_SDF_NOINLINE __attribute__((noinline))
#else
    #define V2M_SDF_NOINLINE inline
#endif

// a stack entry: a value and its gradient
typedef struct {
    float v, x, y, z;
} v2m_sdf_v;

// min (sgn 1) or max (sgn -1) of a and b, crease-blended (radius k) and gaps
// closed (radius kgap; near set if a gap might need the gradients to tell)
inline v2m_sdf_v v2m_sdf_minmax(v2m_sdf_v a, v2m_sdf_v b, float sgn, float k, float kgap, int grad, int* near) {
    const float va = sgn * a.v, vb = sgn * b.v;   // max(a, b) = -min(-a, -b)
    // (a sliver: inside both, thinner than kgap)
    const int sliver = kgap > 0.0f && va > -kgap && vb > -kgap && va + vb < kgap;

    if (sliver && !grad) {
        *near = 1;
    }

    float pen = 0.0f, dpen = 0.0f;   // the penalty, and its derivative factor

    if (sliver && grad) {
        // a gap: the two surfaces facing each other (opposite gradients),
        // closer than an element can resolve -- a sliver of one region
        // between two others, down to a tangent contact of zero
        // thickness. It is closed: a penalty kgap (1 - t / kgap)^2 on
        // its thickness t (full at a contact, none at kgap: about
        // 0.4 kgap closes), so the regions round it meet with a
        // clear interface. Where the surfaces cross at an angle (a
        // junction) nothing changes. (Its gradient, along grad a +
        // grad b, is small inside a sliver but not at its rim, where
        // the junction it makes lies: kept)
        const float ga = sqrt(a.x * a.x + a.y * a.y + a.z * a.z), gb = sqrt(b.x * b.x + b.y * b.y + b.z * b.z);

        if (ga > 0.0f && gb > 0.0f) {
            const float c = (a.x * b.x + a.y * b.y + a.z * b.z) / (ga * gb);
            const float wg = fmin(1.0f, fmax(0.0f, (-c - 0.7f) / 0.25f));
            const float u = fmax(0.0f, 1.0f - (va + vb) / kgap);
            pen = kgap * wg * u * u;
            dpen = 2.0f * wg * u;   // grad (-sgn pen) = dpen (grad a + grad b) (wg's own left out)
        }
    }

    v2m_sdf_v r = a;

    if (k > 0.0f && fabs(va - vb) < k) {
        // quadratic smooth min: C1, within k of the crease only; its
        // gradient is h grad a + (1 - h) grad b exactly
        const float h = 0.5f + 0.5f * (vb - va) / k;
        r.v = sgn * (vb * (1.0f - h) + va * h - k * h * (1.0f - h));
        r.x = h * a.x + (1.0f - h) * b.x;
        r.y = h * a.y + (1.0f - h) * b.y;
        r.z = h * a.z + (1.0f - h) * b.z;
    } else if (vb < va) {
        r = b;
    }

    r.v -= sgn * pen;

    if (dpen > 0.0f) {
        r.x += dpen * (a.x + b.x);
        r.y += dpen * (a.y + b.y);
        r.z += dpen * (a.z + b.z);
    }

    return r;
}

// label l's value s_l at grid-mm point p, and (wantg) its gradient -- returned
// by value: a gradient written through a pointer would keep each caller's
// array in the device's slow local memory
V2M_SDF_NOINLINE v2m_sdf_v v2m_sdf_evalv(V2M_G const float* prog, int l, float px, float py, float pz, int wantg) {
    const int N = (int)prog[0];
    const v2m_sdf_v zero = { 0.0f, 0.0f, 0.0f, 0.0f };
    V2M_SDF_COUNT(wantg);

    if (l < 0 || l >= N) {
        v2m_sdf_v r = zero;
        r.v = -1e30f;
        return r;
    }

    const float x = px + prog[2], y = py + prog[3], z = pz + prog[4];
    // the code: the point's brick's (only the objects near it), else the scene's
    int start = (int)prog[5 + l];
    {
        const int T = (int)prog[8 + N];

        if (T > 0) {
            const int nbx = (int)prog[T], nby = (int)prog[T + 1], nbz = (int)prog[T + 2];
            const float vx = prog[T + 3];
            int bi = (int)floor(px / vx + 0.5f) >> 3, bj = (int)floor(py / vx + 0.5f) >> 3, bk = (int)floor(pz / vx + 0.5f) >> 3;
            bi = bi < 0 ? 0 : (bi >= nbx ? nbx - 1 : bi);
            bj = bj < 0 ? 0 : (bj >= nby ? nby - 1 : bj);
            bk = bk < 0 ? 0 : (bk >= nbz ? nbz - 1 : bk);
            start = (int)prog[(int)prog[T + 4 + bi + nbx * (bj + nby * bk)] + l];
        }
    }

    // the stack: its top 4 in named slots t0 (the top) .. t3, which a push or
    // pop shifts -- in registers (an indexed array would live in the device's
    // slow local memory); the rest, below, in spill (rare: a deep CSG tree)
    v2m_sdf_v t0 = zero, t1 = zero, t2 = zero, t3 = zero, spill[V2M_SDF_STACK - 4];
    int sp = 0, pc = start;
    const float kgap = prog[6 + N], kblend = prog[5 + N];   // the gap-closing (0: off; it needs the gradients) and crease blend radii
    // pass 0: values only (unless g asks); pass 1, only if some min / max had its
    // operands within kgap (a possible gap): again, with the gradients
    int grad = wantg != 0, near = 0;

    for (int pass = 0; pass < 2; ++pass) {
        sp = 0;
        pc = start;
        near = 0;

        for (int guard = 0; guard < 100000; ++guard) {
            const int op = (int)prog[pc];

            if (op == V2M_SDF_END) {
                break;
            }

            if (op == V2M_SDF_MAX || op == V2M_SDF_MIN) {   // t1 op t0 -> t0, pop
                if (sp > 1) {
                    t0 = v2m_sdf_minmax(t1, t0, op == V2M_SDF_MAX ? -1.0f : 1.0f, kblend, kgap, grad, &near);
                    t1 = t2;
                    t2 = t3;

                    if (sp > 4) {
                        t3 = spill[sp - 5];
                    }

                    --sp;
                }

                pc += 1;
                continue;
            }

            if (op == V2M_SDF_NEG) {
                t0.v = -t0.v;
                t0.x = -t0.x;
                t0.y = -t0.y;
                t0.z = -t0.z;
                pc += 1;
                continue;
            }

            if (op == V2M_SDF_ADD) {   // t1 + t0 -> t0, pop
                if (sp > 1) {
                    t0.v += t1.v;
                    t0.x += t1.x;
                    t0.y += t1.y;
                    t0.z += t1.z;
                    t1 = t2;
                    t2 = t3;

                    if (sp > 4) {
                        t3 = spill[sp - 5];
                    }

                    --sp;
                }

                pc += 1;
                continue;
            }

            if (op == V2M_SDF_SCALE) {
                const float f = prog[pc + 1];
                t0.v *= f;
                t0.x *= f;
                t0.y *= f;
                t0.z *= f;
                pc += 2;
                continue;
            }

            v2m_sdf_v nv = zero;   // an operand: combined at once, or pushed

            if (op == V2M_SDF_PRIM) {
                const int type = (int)prog[pc + 1];
                V2M_G const float* q = prog + pc + 2;
                // the value, and (grad) central differences of this primitive
                // (step 1e-4 of its scale, >= 1e-5): one call site in a loop --
                // the device compiler inlines it once, not 7 times (and fv is best
                // left an array: kept in registers instead, k_move took 30% longer)
                const float e = fmax(1e-4f * (fabs(q[0]) + fabs(q[1]) + fabs(q[2]) + 1.0f), 1e-5f);
                const int ne = grad ? 7 : 1;
                float fv[7];

                for (int k = 0; k < ne; ++k) {
                    const float dd = k == 0 ? 0.0f : ((k & 1) ? e : -e);
                    const int ax = (k - 1) >> 1;
                    fv[k] = v2m_sdf_prim(type, q, x + (ax == 0 ? dd : 0.0f), y + (ax == 1 ? dd : 0.0f), z + (ax == 2 ? dd : 0.0f));
                }

                nv.v = fv[0];

                if (grad) {
                    nv.x = (fv[1] - fv[2]) / (2.0f * e);
                    nv.y = (fv[3] - fv[4]) / (2.0f * e);
                    nv.z = (fv[5] - fv[6]) / (2.0f * e);
                }

                pc += 2 + v2m_sdf_nparam(type);
            } else if (op == V2M_SDF_BBOX) {
                // bounding-box culling: an object's term far from the point is
                // pushed as the constant (a bound its real value is beyond, where
                // it cannot decide anything) and its code skipped
                const float m = prog[7 + N];
                V2M_G const float* q = prog + pc + 1;

                if (!(x < q[0] - m || y < q[1] - m || z < q[2] - m || x > q[3] + m || y > q[4] + m || z > q[5] + m)) {
                    pc += 9;
                    continue;
                }

                nv.v = q[7] * m;
                pc += 9 + (int)q[6];
            } else if (op == V2M_SDF_MCONST) {
                nv.v = prog[pc + 1] * prog[7 + N];
                pc += 2;
            } else if (op == V2M_SDF_CONST) {
                nv.v = prog[pc + 1];
                pc += 2;
            } else {
                break;   // (a malformed program)
            }

            // an operand combined at once (the usual case: "x MIN", "x NEG MIN" --
            // an accumulator), without the stack's shifts (the second call site
            // of v2m_sdf_minmax: sharing one made k_move 30% slower)
            {
                const int op1 = (int)prog[pc], neg = op1 == V2M_SDF_NEG;
                const int op2 = neg ? (int)prog[pc + 1] : op1;

                if (sp > 0 && (op2 == V2M_SDF_MAX || op2 == V2M_SDF_MIN)) {
                    if (neg) {
                        nv.v = -nv.v;
                        nv.x = -nv.x;
                        nv.y = -nv.y;
                        nv.z = -nv.z;
                    }

                    t0 = v2m_sdf_minmax(t0, nv, op2 == V2M_SDF_MAX ? -1.0f : 1.0f, kblend, kgap, grad, &near);
                    pc += 1 + neg;
                    continue;
                }
            }

            if (sp < V2M_SDF_STACK) {   // push
                if (sp >= 4) {
                    spill[sp - 4] = t3;
                }

                t3 = t2;
                t2 = t1;
                t1 = t0;
                t0 = nv;
                ++sp;
            }
        }

        if (grad || !near) {
            break;
        }

        grad = 1;
    }

    if (sp > 0) {
        return t0;
    }

    t0 = zero;
    t0.v = -1e30f;
    return t0;
}

// the same, the gradient (if g) through a pointer (the host's callers)
inline float v2m_sdf_eval(V2M_G const float* prog, int l, float px, float py, float pz, float* g) {
    const v2m_sdf_v r = v2m_sdf_evalv(prog, l, px, py, pz, g != 0);

    if (g) {
        g[0] = r.x;
        g[1] = r.y;
        g[2] = r.z;
    }

    return r.v;
}
