// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_sdf_body.cl -- analytic signed distance functions of shape constructs
// (JSON shapes, tn_sdfshape.h): one postfix program per label, evaluated with
// a fixed stack. Shared by the host (TN_G empty) and the OpenCL device.
//
// Program layout (floats):
//   [0] N labels   [1] w (the field width: phi = clamp(0.5 + s / w))
//   [2..4] origin (world = grid mm + origin)   [5 .. 5+N-1] code offsets
//   [5+N] the blend radius of min / max (0: exact)
//   code: opcode, operands ...; TN_SDF_END ends a label's code
// Values are s > 0 inside, world units; each primitive's gradient comes along
// (analytic, or central differences of that primitive alone), carried through
// max / min by the winning branch.

#define TN_SDF_END 0
#define TN_SDF_PRIM 1     // type, parameters
#define TN_SDF_MAX 2      // union
#define TN_SDF_MIN 3      // intersection
#define TN_SDF_NEG 4      // complement
#define TN_SDF_CONST 5    // value

#define TN_SDF_SPHERE 1     // c[3] r
#define TN_SDF_BOX 2        // lo[3] hi[3] (axis-aligned)
#define TN_SDF_CYL 3        // a[3] b[3] r (capped)
#define TN_SDF_PLANE 4      // o[3] n[3] (unit): the half-space behind n
#define TN_SDF_SLAB 5       // o[3] n[3] lo hi: lo < (p - o).n < hi
#define TN_SDF_TORUS 6      // c[3] n[3] R r
#define TN_SDF_ELLIPSOID 7  // c[3] r[3] rot[9] (row-major, world -> local)
#define TN_SDF_CONE 8       // a[3] b[3] ra rb (capped frustum)
#define TN_SDF_INFCYL 9     // o[3] n[3] r (infinite)

#define TN_SDF_STACK 32

inline int tn_sdf_nparam(int type) {
    switch (type) {
    case TN_SDF_SPHERE:
        return 4;

    case TN_SDF_BOX:
        return 6;

    case TN_SDF_CYL:
    case TN_SDF_INFCYL:
        return 7;

    case TN_SDF_PLANE:
        return 6;

    case TN_SDF_SLAB:
    case TN_SDF_TORUS:
    case TN_SDF_CONE:
        return 8;

    case TN_SDF_ELLIPSOID:
        return 15;

    default:
        return 0;
    }
}

// one primitive's value at (x, y, z) (world), s > 0 inside
inline float tn_sdf_prim(int type, TN_G const float* q, float x, float y, float z) {
    if (type == TN_SDF_SPHERE) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        return q[3] - sqrt(dx * dx + dy * dy + dz * dz);
    }

    if (type == TN_SDF_BOX) {   // exact box distance
        const float cx = 0.5f * (q[0] + q[3]), cy = 0.5f * (q[1] + q[4]), cz = 0.5f * (q[2] + q[5]);
        const float ux = fabs(x - cx) - 0.5f * fabs(q[3] - q[0]), uy = fabs(y - cy) - 0.5f * fabs(q[4] - q[1]),
                    uz = fabs(z - cz) - 0.5f * fabs(q[5] - q[2]);
        const float ox = fmax(ux, 0.0f), oy = fmax(uy, 0.0f), oz = fmax(uz, 0.0f);
        const float out = sqrt(ox * ox + oy * oy + oz * oz), in = fmin(fmax(ux, fmax(uy, uz)), 0.0f);
        return -(out + in);
    }

    if (type == TN_SDF_PLANE) {
        return -((x - q[0]) * q[3] + (y - q[1]) * q[4] + (z - q[2]) * q[5]);
    }

    if (type == TN_SDF_SLAB) {
        const float t = (x - q[0]) * q[3] + (y - q[1]) * q[4] + (z - q[2]) * q[5];
        return fmin(t - q[6], q[7] - t);
    }

    if (type == TN_SDF_INFCYL) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        const float t = dx * q[3] + dy * q[4] + dz * q[5];
        const float rx = dx - t * q[3], ry = dy - t * q[4], rz = dz - t * q[5];
        return q[6] - sqrt(rx * rx + ry * ry + rz * rz);
    }

    if (type == TN_SDF_TORUS) {
        const float dx = x - q[0], dy = y - q[1], dz = z - q[2];
        const float h = dx * q[3] + dy * q[4] + dz * q[5];
        const float rx = dx - h * q[3], ry = dy - h * q[4], rz = dz - h * q[5];
        const float a = sqrt(rx * rx + ry * ry + rz * rz) - q[6];
        return q[7] - sqrt(a * a + h * h);
    }

    if (type == TN_SDF_CYL) {   // capped cylinder (exact; I. Quilez's formulation)
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

    if (type == TN_SDF_CONE) {   // capped frustum a (radius ra) .. b (radius rb) (exact; I. Quilez)
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

    if (type == TN_SDF_ELLIPSOID) {   // a bound (exact on the surface): k0 (k0 - 1) / k1
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

// label l's value s_l at grid-mm point p, and its gradient g (may be 0)
inline float tn_sdf_eval(TN_G const float* prog, int l, float px, float py, float pz, float* g) {
    const int N = (int)prog[0];

    if (l < 0 || l >= N) {
        if (g) {
            g[0] = g[1] = g[2] = 0.0f;
        }

        return -1e30f;
    }

    const float x = px + prog[2], y = py + prog[3], z = pz + prog[4];
    float sv[TN_SDF_STACK], sg[TN_SDF_STACK * 3];
    int sp = 0, pc = (int)prog[5 + l];

    for (int guard = 0; guard < 100000; ++guard) {
        const int op = (int)prog[pc];

        if (op == TN_SDF_END) {
            break;
        }

        if (op == TN_SDF_PRIM) {
            const int type = (int)prog[pc + 1];
            TN_G const float* q = prog + pc + 2;
            const float v = tn_sdf_prim(type, q, x, y, z);

            if (sp < TN_SDF_STACK) {
                sv[sp] = v;

                if (g) {   // central differences of this primitive (step: 1e-4 of its scale, >= 1e-5)
                    const float e = fmax(1e-4f * (fabs(q[0]) + fabs(q[1]) + fabs(q[2]) + 1.0f), 1e-5f);
                    sg[3 * sp] = (tn_sdf_prim(type, q, x + e, y, z) - tn_sdf_prim(type, q, x - e, y, z)) / (2.0f * e);
                    sg[3 * sp + 1] = (tn_sdf_prim(type, q, x, y + e, z) - tn_sdf_prim(type, q, x, y - e, z)) / (2.0f * e);
                    sg[3 * sp + 2] = (tn_sdf_prim(type, q, x, y, z + e) - tn_sdf_prim(type, q, x, y, z - e)) / (2.0f * e);
                }

                ++sp;
            }

            pc += 2 + tn_sdf_nparam(type);
        } else if (op == TN_SDF_CONST) {
            if (sp < TN_SDF_STACK) {
                sv[sp] = prog[pc + 1];
                sg[3 * sp] = sg[3 * sp + 1] = sg[3 * sp + 2] = 0.0f;
                ++sp;
            }

            pc += 2;
        } else if (op == TN_SDF_NEG) {
            if (sp > 0) {
                sv[sp - 1] = -sv[sp - 1];
                sg[3 * sp - 3] = -sg[3 * sp - 3];
                sg[3 * sp - 2] = -sg[3 * sp - 2];
                sg[3 * sp - 1] = -sg[3 * sp - 1];
            }

            pc += 1;
        } else if (op == TN_SDF_MAX || op == TN_SDF_MIN) {
            if (sp > 1) {
                const int a = sp - 2, b = sp - 1;
                const float k = prog[5 + N];   // the blend radius (0: exact min / max)
                const float sgn = op == TN_SDF_MAX ? -1.0f : 1.0f;   // max(a, b) = -min(-a, -b)
                const float va = sgn * sv[a], vb = sgn * sv[b];

                if (k > 0.0f && fabs(va - vb) < k) {
                    // quadratic smooth min: C1, within k of the crease only; its
                    // gradient is h grad a + (1 - h) grad b exactly
                    const float h = 0.5f + 0.5f * (vb - va) / k;
                    sv[a] = sgn * (vb * (1.0f - h) + va * h - k * h * (1.0f - h));
                    sg[3 * a] = h * sg[3 * a] + (1.0f - h) * sg[3 * b];
                    sg[3 * a + 1] = h * sg[3 * a + 1] + (1.0f - h) * sg[3 * b + 1];
                    sg[3 * a + 2] = h * sg[3 * a + 2] + (1.0f - h) * sg[3 * b + 2];
                } else if (vb < va) {
                    sv[a] = sv[b];
                    sg[3 * a] = sg[3 * b];
                    sg[3 * a + 1] = sg[3 * b + 1];
                    sg[3 * a + 2] = sg[3 * b + 2];
                }

                --sp;
            }

            pc += 1;
        } else {
            break;   // (a malformed program)
        }
    }

    if (g) {
        g[0] = sp > 0 ? sg[0] : 0.0f;
        g[1] = sp > 0 ? sg[1] : 0.0f;
        g[2] = sp > 0 ? sg[2] : 0.0f;
    }

    return sp > 0 ? sv[0] : -1e30f;
}
