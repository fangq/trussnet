// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_del_body.cl -- filtered geometric predicates of the GPU Delaunay, shared by
// the OpenCL kernels (tn_del_kernels.cl) and the host (the sign-convention test).
// These are the floating-point FILTERS of the vendored Diazzi/Attene predicates
// (third_party/cdt: orient3d_filtered, inSphere_filtered), transcribed verbatim so
// that every sign they certify equals the one the exact CPU Delaunay computes.
// They return 0 when the filter cannot certify the sign; the device then DEFERS
// the point to the exact CPU insertion (no exact arithmetic on the device), so
// the result is exactly Diazzi's symbolically-perturbed Delaunay tetrahedrization.
// Must be compiled without FMA contraction (#pragma OPENCL FP_CONTRACT OFF).

#ifndef TN_DEL_BODY_CL
#define TN_DEL_BODY_CL

// orient3d_filtered(p, q, r, s): +1 / -1, 0 = uncertain
inline int tn_o3d_f(const double* p, const double* q, const double* r, const double* s) {
    double fadx = q[0] - p[0], fbdx = r[0] - p[0], fcdx = s[0] - p[0];
    double fady = q[1] - p[1], fbdy = r[1] - p[1], fcdy = s[1] - p[1];
    double fadz = q[2] - p[2], fbdz = r[2] - p[2], fcdz = s[2] - p[2];
    double fbdxcdy = fbdx * fcdy * fadz, fcdxbdy = fcdx * fbdy * fadz;
    double fcdxady = fcdx * fady * fbdz, fadxcdy = fadx * fcdy * fbdz;
    double fadxbdy = fadx * fbdy * fcdz, fbdxady = fbdx * fady * fcdz;
    double det = (fbdxcdy - fcdxbdy) + (fcdxady - fadxcdy) + (fadxbdy - fbdxady);
    double eb = 7.7715611723761027e-016 * (fabs(fbdxcdy) + fabs(fcdxbdy) + fabs(fcdxady) + fabs(fadxcdy) +
                                           fabs(fadxbdy) + fabs(fbdxady));
    return (det >= eb) - (-det >= eb);
}

// inSphere_filtered(a, b, c, d, e): +1 / -1, 0 = uncertain
inline int tn_isp_f(const double* pa, const double* pb, const double* pc, const double* pd, const double* pe) {
    const double aex = pa[0] - pe[0], aey = pa[1] - pe[1], aez = pa[2] - pe[2];
    const double bex = pb[0] - pe[0], bey = pb[1] - pe[1], bez = pb[2] - pe[2];
    const double cex = pc[0] - pe[0], cey = pc[1] - pe[1], cez = pc[2] - pe[2];
    const double dex = pd[0] - pe[0], dey = pd[1] - pe[1], dez = pd[2] - pe[2];
    const double aexbey = aex * bey, bexaey = bex * aey, ab = aexbey - bexaey;
    const double bexcey = bex * cey, cexbey = cex * bey, bc = bexcey - cexbey;
    const double cexdey = cex * dey, dexcey = dex * cey, cd = cexdey - dexcey;
    const double dexaey = dex * aey, aexdey = aex * dey, da = dexaey - aexdey;
    const double aexcey = aex * cey, cexaey = cex * aey, ac = aexcey - cexaey;
    const double bexdey = bex * dey, dexbey = dex * bey, bd = bexdey - dexbey;
    const double abc1 = aez * bc, abc2 = bez * ac, abc3 = cez * ab;
    const double abc4 = abc1 + abc3, abc = abc4 - abc2;
    const double bcd1 = bez * cd, bcd2 = cez * bd, bcd3 = dez * bc;
    const double bcd4 = bcd1 + bcd3, bcd = bcd4 - bcd2;
    const double cda1 = cez * da, cda2 = dez * ac, cda3 = aez * cd;
    const double cda4 = cda1 + cda3, cda = cda4 + cda2;
    const double dab1 = dez * ab, dab2 = aez * bd, dab3 = bez * da;
    const double dab4 = dab1 + dab3, dab = dab4 + dab2;
    const double al1 = aex * aex, al2 = aey * aey, al3 = aez * aez;
    const double al4 = al1 + al2, alift = al4 + al3;
    const double bl1 = bex * bex, bl2 = bey * bey, bl3 = bez * bez;
    const double bl4 = bl1 + bl2, blift = bl4 + bl3;
    const double cl1 = cex * cex, cl2 = cey * cey, cl3 = cez * cez;
    const double cl4 = cl1 + cl2, clift = cl4 + cl3;
    const double dl1 = dex * dex, dl2 = dey * dey, dl3 = dez * dez;
    const double dl4 = dl1 + dl2, dlift = dl4 + dl3;
    const double ds1 = dlift * abc, ds2 = clift * dab, dl = ds2 - ds1;
    const double dr1 = blift * cda, dr2 = alift * bcd, dr = dr2 - dr1;
    const double det = dl + dr;
    double m = 0.0, t;

    if ((t = fabs(aex)) > m) m = t;
    if ((t = fabs(aey)) > m) m = t;
    if ((t = fabs(aez)) > m) m = t;
    if ((t = fabs(bex)) > m) m = t;
    if ((t = fabs(bey)) > m) m = t;
    if ((t = fabs(bez)) > m) m = t;
    if ((t = fabs(cex)) > m) m = t;
    if ((t = fabs(cey)) > m) m = t;
    if ((t = fabs(cez)) > m) m = t;
    if ((t = fabs(dex)) > m) m = t;
    if ((t = fabs(dey)) > m) m = t;
    if ((t = fabs(dez)) > m) m = t;

    double eps = m;
    eps *= eps;
    eps *= eps;
    eps *= m;
    eps *= 1.145750161413163e-13;

    if (det > eps) {
        return 1;
    }

    if (-det > eps) {
        return -1;
    }

    return 0;
}

#endif  // TN_DEL_BODY_CL
