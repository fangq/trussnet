// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_del_kernels.cl -- device-resident Delaunay tetrahedrization by rounds of
// parallel Bowyer-Watson insertion, on the data structure of the vendored Diazzi
// TetMesh so the result loads into it unchanged:
//   node[4t..4t+3]  corners (ghost tets: node[4t+3] = INF, the infinite vertex)
//   neigh[4t+i]     the CORNER 4t'+i' of the tet across face i (opposite node i)
// A round:
//   d_locate   every pending point walks (visibility walk) to its tet; the tet
//              keeps the smallest priority of the points inside (atomic_min)
//   d_pick     the point holding its tet's minimum becomes a candidate
//   d_cavity   each candidate grows its conflict region (exact-sign filters;
//              an uncertain sign DEFERS the point to the exact CPU insertion)
//   d_claim / d_check / d_fix / d_clear, a few passes: an independent set of
//              non-conflicting cavities by priority (Luby); winners reserve the
//              slots their cavity lacks (#boundary faces - #cavity tets)
//   d_commit   winners re-tetrahedrize (the star of the new point, Diazzi's
//              corner order), reusing their own cavity slots
//   d_reset    candidates release their claims
// Priorities are a bijective hash of the point id, so the insertion order is a
// pseudo-random one and the global minimum candidate always wins (progress).
// Built with: tn_del_body.cl + this file; FP contraction off.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

#define INF 0xFFFFFFFFu
#define DEAD 0xFFFFFFFEu
#define NONE 0xFFFFFFFFu
#define MAXC 128   // cavity tets per candidate
#define MAXB 256   // boundary faces per candidate
#define MAXWALK 512
#ifndef HBITS
#define HBITS 9    // the cavity's seen-set: up to 512 slots, at most half full
#endif
#define HSZ (1u << HBITS)
#define EBITS 9    // the commit's edge table: 512 slots (<= 3 MAXB / 2 = 384 edges)
#define ESZ (1u << EBITS)

// point states
#define ST_PENDING 0
#define ST_INSERTED 1
#define ST_DEFER 2

// 1 + an odd-multiplier hash mod 2^31: a bijection for p < 2^31, never 0 (the
// "fixed" mark of the claims)
inline uint prio_of(uint p) {
    return 1u + ((p * 2654435761u) & 0x7FFFFFFFu);
}

inline void ld(__global const double* X, uint v, double* o) {
    o[0] = X[3 * (size_t)v];
    o[1] = X[3 * (size_t)v + 1];
    o[2] = X[3 * (size_t)v + 2];
}

// Diazzi's vOrient3D(a, b, c, d) = -orient3D(a, b, c, d); 0 = uncertain
inline int vo3d(__global const double* X, uint a, uint b, uint c, uint d) {
    double pa[3], pb[3], pc[3], pd[3];
    ld(X, a, pa);
    ld(X, b, pb);
    ld(X, c, pc);
    ld(X, d, pd);
    return -tn_o3d_f(pa, pb, pc, pd);
}

// Diazzi's vertexInTetSphere(t, p): > 0 in conflict, < 0 not, 0 = uncertain.
// (A certified sign never needs the symbolic perturbation: that is only for an
// exact zero, which the filter cannot distinguish from "close" -> deferred.)
inline int in_sphere(__global const double* X, __global const uint* node, __global const uint* neigh, uint t,
                     uint p) {
    const uint a = node[4 * t], b = node[4 * t + 1], c = node[4 * t + 2], d = node[4 * t + 3];

    if (d == INF) {
        return vo3d(X, a, b, c, p);   // an exact 0 would need the real neighbour's sphere: deferred
    }

    double pa[3], pb[3], pc[3], pd[3], pe[3];
    ld(X, a, pa);
    ld(X, b, pb);
    ld(X, c, pc);
    ld(X, d, pd);
    ld(X, p, pe);
    return -tn_isp_f(pa, pb, pc, pd, pe);
}

__kernel void d_locate(__global const double* X, __global const uint* node, __global const uint* neigh,
                       __global uint* loc, __global const uchar* state, __global uint* pick, __global uint* stats,
                       uint n) {
    const uint p = get_global_id(0);

    if (p >= n || state[p] != ST_PENDING) {
        return;
    }

    uint t = loc[p];

    while (node[4 * t] == DEAD) {   // a slot freed by a cavity: follow its forward pointer
        t = neigh[4 * t] >> 2;
    }

    if (node[4 * t + 3] == INF) {
        t = neigh[4 * t + 3] >> 2;
    }

    uint f0 = 4, steps = 0;
    bool done = false;

    while (steps++ < MAXWALK) {
        const uint N0 = node[4 * t], N1 = node[4 * t + 1], N2 = node[4 * t + 2], N3 = node[4 * t + 3];

        if (N3 == INF) {
            done = true;
            break;
        }

        // faces i: (ON1(i), ON2(i), ON3(i)) = (1,3,2) (2,3,0) (3,1,0) (0,1,2)
        uint i;

        for (i = 0; i < 4; ++i) {
            if (i == f0) {
                continue;
            }

            int o;

            if (i == 0) {
                o = vo3d(X, N1, N3, N2, p);
            } else if (i == 1) {
                o = vo3d(X, N2, N3, N0, p);
            } else if (i == 2) {
                o = vo3d(X, N3, N1, N0, p);
            } else {
                o = vo3d(X, N0, N1, N2, p);
            }

            if (o < 0) {   // (an uncertain face is not crossed; d_cavity verifies the start)
                const uint ni = neigh[4 * t + i];
                t = ni >> 2;
                f0 = ni & 3;
                break;
            }
        }

        if (i == 4) {
            done = true;
            break;
        }
    }

    loc[p] = t;

    if (done) {
        atomic_min(&pick[t], prio_of(p));
    }

    atomic_inc(&stats[0]);   // pending points
}

// A point is a candidate if it holds its tet's minimum and (filter) that minimum is
// also below the minima of the 4 face-adjacent tets: a winner needs the lowest
// priority among the cavities around it anyway, and this skips computing most of
// the cavities that would lose.
__kernel void d_pick(__global const uint* loc, __global const uchar* state, __global const uint* pick,
                     __global const uint* neigh, __global uint* cand, __global uint* stats, uint n, uint maxcand,
                     uint filter) {
    const uint p = get_global_id(0);

    if (p >= n || state[p] != ST_PENDING) {
        return;
    }

    const uint t = loc[p], pr = prio_of(p);

    if (pick[t] != pr) {
        return;
    }

    if (filter) {   // 1: the 4 face neighbours; 2: and theirs
        for (uint f = 0; f < 4; ++f) {
            const uint u = neigh[4 * t + f] >> 2;

            if (pick[u] < pr) {
                return;
            }

            if (filter > 1) {
                for (uint g = 0; g < 4; ++g) {
                    const uint w = neigh[4 * u + g] >> 2;

                    if (pick[w] < pr) {
                        return;
                    }

                    if (filter > 2) {
                        for (uint h = 0; h < 4; ++h) {
                            if (pick[neigh[4 * w + h] >> 2] < pr) {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    const uint k = atomic_inc(&stats[1]);

    if (k < maxcand) {
        cand[k] = p;
    }
}

__kernel void d_unpick(__global const uint* loc, __global const uchar* state, __global uint* pick, uint n) {
    const uint p = get_global_id(0);

    if (p < n) {
        pick[loc[p]] = NONE;   // (every point: the ones inserted this round too)
    }
}

inline bool in_list(__global const uint* L, uint m, uint t) {
    for (uint k = 0; k < m; ++k) {
        if (L[k] == t) {
            return true;
        }
    }

    return false;
}

// candidate states (cst)
#define C_UNDEC 0
#define C_WIN 1
#define C_OUT 2
#define C_NEWWIN 3

// cavT[MAXC*ci ..] cavity tets; cavB[MAXB*ci ..] boundary corners (the corner of the
// OUTSIDE tet across each boundary face, as Diazzi's cavityCorners); cnt[2ci] = #tets,
// cnt[2ci+1] = #faces. No claims here: the lists are computed once per round
// against the unchanged mesh, then several selection passes run on them.
__kernel void d_cavity(__global const double* X, __global const uint* node, __global const uint* neigh,
                       __global const uint* loc, __global uchar* state, __global uchar* ovf, __global const uint* cand,
                       __global uint* cavT, __global uint* cavB, __global uint* cnt, __global uchar* cst,
                       __global uint* stats, uint maxcand) {
    const uint ci = get_global_id(0);
    const uint nc_all = min(stats[1], maxcand);

    if (ci >= nc_all) {
        return;
    }

    const uint p = cand[ci];
    const uint t0 = loc[p];
    __global uint* T = cavT + (size_t)MAXC * ci;
    __global uint* B = cavB + (size_t)MAXB * ci;
    cnt[2 * ci] = 0;
    cnt[2 * ci + 1] = 0;
    cst[ci] = C_OUT;
    const int s0 = in_sphere(X, node, neigh, t0, p);

    if (s0 <= 0) {   // uncertain start (point on/near a face or a sphere): exact CPU insertion
        state[p] = ST_DEFER;
        atomic_inc(&stats[3]);
        return;
    }

    // the tets seen so far (cavity or rejected), open addressing: key t | bit31 =
    // rejected. It starts at 64 slots (clearing all HSZ per candidate cost more than
    // the search) and is rehashed once to HSZ when half full.
    uint seen[HSZ + 64];
    uint hm = 64, hb = 6;

    for (uint k = 0; k < 64; ++k) {
        seen[k] = NONE;
    }

    uint nt = 0, nb = 0, ns = 0;
    T[nt++] = t0;
    seen[(t0 * 2654435761u) >> (32 - hb)] = t0;
    ns = 1;

    for (uint i = 0; i < nt; ++i) {
        const uint t = T[i];

        for (uint f = 0; f < 4; ++f) {
            const uint c = neigh[4 * t + f];
            const uint u = c >> 2;
            uint h = (u * 2654435761u) >> (32 - hb);

            while (seen[h] != NONE && (seen[h] & 0x7FFFFFFFu) != u) {
                h = (h + 1) & (hm - 1);
            }

            if (seen[h] == NONE) {   // first visit: test it
                if (ns == hm / 2) {
                    if (hm == HSZ) {
                        goto overflow;
                    }

                    for (uint k = 0; k < 64; ++k) {   // grow: park the 64 old slots past HSZ
                        seen[HSZ + k] = seen[k];
                    }

                    hm = HSZ;
                    hb = HBITS;

                    for (uint k = 0; k < HSZ; ++k) {
                        seen[k] = NONE;
                    }

                    for (uint k = 0; k < 64; ++k) {
                        const uint e = seen[HSZ + k];

                        if (e != NONE) {
                            uint g = ((e & 0x7FFFFFFFu) * 2654435761u) >> (32 - hb);

                            while (seen[g] != NONE) {
                                g = (g + 1) & (hm - 1);
                            }

                            seen[g] = e;
                        }
                    }

                    h = (u * 2654435761u) >> (32 - hb);

                    while (seen[h] != NONE) {
                        h = (h + 1) & (hm - 1);
                    }
                }

                const int sg = in_sphere(X, node, neigh, u, p);

                if (sg == 0) {
                    state[p] = ST_DEFER;
                    atomic_inc(&stats[3]);
                    return;
                }

                ++ns;

                if (sg > 0) {
                    if (nt == MAXC) {
                        goto overflow;
                    }

                    seen[h] = u;
                    T[nt++] = u;
                    continue;
                }

                seen[h] = u | 0x80000000u;
            } else if (!(seen[h] & 0x80000000u)) {
                continue;   // in the cavity
            }

            if (nb == MAXB) {
                goto overflow;
            }

            B[nb++] = c;
        }
    }

    cnt[2 * ci] = nt;
    cnt[2 * ci + 1] = nb;
    cst[ci] = C_UNDEC;
    return;
overflow:

    if (++ovf[p] >= 3) {   // persistently huge (early far-outside points): exact CPU insertion
        state[p] = ST_DEFER;
        atomic_inc(&stats[3]);
    }

    atomic_inc(&stats[4]);
}

// Conflicts: A and B conflict iff cav(A) meets cav(B), or cav(A) meets the shell
// (the tets across the boundary faces) of B. Two shells may share a tet: each
// rewrites only its own face of it. Claims: ownC (cavity) / ownS (shell) take the
// minimum priority; 0 marks the tets of the winners fixed in an earlier pass.
// A pass is claim -> check -> fix -> clear (Luby: winners are the local minima of
// the undecided ones; the ones they block drop out).
__kernel void d_claim(__global const uint* cand, __global const uint* cavT, __global const uint* cavB,
                      __global const uint* cnt, __global uchar* cst, __global uint* ownC, __global uint* ownS,
                      __global const uint* stats, uint maxcand) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand) || cst[ci] != C_UNDEC) {
        return;
    }

    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1];
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;

    for (uint i = 0; i < nt; ++i) {
        if (ownC[T[i]] == 0 || ownS[T[i]] == 0) {
            cst[ci] = C_OUT;
            return;
        }
    }

    for (uint i = 0; i < nb; ++i) {
        if (ownC[B[i] >> 2] == 0) {
            cst[ci] = C_OUT;
            return;
        }
    }

    const uint pr = prio_of(cand[ci]);

    for (uint i = 0; i < nt; ++i) {
        if (ownC[T[i]] > pr) {   // (a plain read first: most claims are already beaten)
            atomic_min(&ownC[T[i]], pr);
        }
    }

    for (uint i = 0; i < nb; ++i) {
        if (ownS[B[i] >> 2] > pr) {
            atomic_min(&ownS[B[i] >> 2], pr);
        }
    }
}

__kernel void d_check(__global const uint* cand, __global const uint* cavT, __global const uint* cavB,
                      __global const uint* cnt, __global uchar* cst, __global const uint* ownC,
                      __global const uint* ownS, __global const uint* stats, uint maxcand) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand) || cst[ci] != C_UNDEC) {
        return;
    }

    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1];
    const uint pr = prio_of(cand[ci]);
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;

    for (uint i = 0; i < nt; ++i) {
        if (ownC[T[i]] != pr || ownS[T[i]] < pr) {
            return;
        }
    }

    for (uint i = 0; i < nb; ++i) {
        if (ownC[B[i] >> 2] < pr) {
            return;
        }
    }

    cst[ci] = C_NEWWIN;
}

// new winners: fix their tets (0) and reserve the slots their cavity lacks
// (#boundary faces - #cavity tets); win[ci] = the first fresh slot
__kernel void d_fix(__global const uint* cavT, __global const uint* cavB, __global const uint* cnt,
                    __global uchar* cst, __global uint* ownC, __global uint* ownS, __global uint* win,
                    __global uint* stats, uint maxcand, uint cap) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand) || cst[ci] != C_NEWWIN) {
        return;
    }

    cst[ci] = C_WIN;
    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1];
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;

    for (uint i = 0; i < nt; ++i) {
        ownC[T[i]] = 0;
    }

    for (uint i = 0; i < nb; ++i) {
        ownS[B[i] >> 2] = 0;
    }

    uint base = 0;

    if (nb > nt) {
        base = atomic_add(&stats[2], nb - nt);

        if (base + (nb - nt) > cap) {
            atomic_inc(&stats[5]);   // out of slots: the host grows the arrays and redoes the round
        }
    }

    win[ci] = base;
    atomic_inc(&stats[6]);
}

// release the claims of the ones still undecided (the winners' are 0 now; the ones
// out made none this pass)
__kernel void d_clear(__global const uint* cavT, __global const uint* cavB, __global const uint* cnt,
                      __global const uchar* cst, __global uint* ownC, __global uint* ownS,
                      __global const uint* stats, uint maxcand) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand) || cst[ci] != C_UNDEC) {
        return;
    }

    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1];
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;

    for (uint i = 0; i < nt; ++i) {
        if (ownC[T[i]] != 0) {
            ownC[T[i]] = NONE;
        }
    }

    for (uint i = 0; i < nb; ++i) {
        if (ownS[B[i] >> 2] != 0) {
            ownS[B[i] >> 2] = NONE;
        }
    }
}

__kernel void d_commit(__global const uint* cand, __global const uint* cavT, __global const uint* cavB,
                       __global const uint* cnt, __global const uchar* cst, __global const uint* win,
                       __global uint* node, __global uint* neigh, __global uchar* state, __global uint* stats,
                       uint maxcand) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand) || cst[ci] != C_WIN || stats[5]) {
        return;
    }

    const uint p = cand[ci];
    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1], base = win[ci];
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;
    // Diazzi: new tet = (p, cr[fi[cb][0..2]]) across the boundary corner c = 4*u + cb
    const uchar fi[12] = { 2, 1, 3, 0, 2, 3, 1, 0, 3, 0, 1, 2 };
#define SLOT(k) ((k) < nt ? T[k] : base + ((k) - nt))

    for (uint k = 0; k < nb; ++k) {
        const uint c = B[k], cb = c & 3, cr = c - cb, s = SLOT(k);
        const uint a = node[cr + fi[3 * cb]], b = node[cr + fi[3 * cb + 1]], d = node[cr + fi[3 * cb + 2]];
        node[4 * s] = p;
        node[4 * s + 1] = a;
        node[4 * s + 2] = b;
        node[4 * s + 3] = d;
        neigh[4 * s] = c;
        neigh[c] = 4 * s;
    }

    // internal adjacency: face f (1..3) of new tet k is (p, the two corners != f), i.e.
    // the boundary edge {e0, e1}; its two new tets meet across it. Edge hash: the
    // first one to arrive parks (edge, k, f), the second links both.
    uint ek[2 * ESZ];   // ESZ slots: edge (e0, e1) -> ev, the parked corner
    uint ev[ESZ];

    uint eb = 4;   // table: the power of 2 >= 3 nb (<= 2x the 3 nb / 2 edges)

    while ((1u << eb) < 3 * nb && eb < EBITS) {
        ++eb;
    }

    const uint em = 1u << eb;

    for (uint h = 0; h < em; ++h) {
        ev[h] = NONE;
    }

    for (uint k = 0; k < nb; ++k) {
        const uint s = SLOT(k);
        const uint v[3] = { node[4 * s + 1], node[4 * s + 2], node[4 * s + 3] };

        for (uint f = 1; f <= 3; ++f) {
            const uint e0 = min(v[f % 3], v[(f + 1) % 3]), e1 = max(v[f % 3], v[(f + 1) % 3]);
            uint h = ((e0 * 2654435761u) ^ (e1 * 40503u)) >> (32 - eb);

            while (ev[h] != NONE && (ek[2 * h] != e0 || ek[2 * h + 1] != e1)) {
                h = (h + 1) & (em - 1);
            }

            if (ev[h] == NONE) {
                ek[2 * h] = e0;
                ek[2 * h + 1] = e1;
                ev[h] = 4 * s + f;   // this corner
            } else {
                const uint o = ev[h];   // the partner corner 4*s'+f'
                neigh[4 * s + f] = o;
                neigh[o] = 4 * s + f;
                ev[h] = NONE - 1;       // consumed (kept non-empty so probing continues past it)
                ek[2 * h] = NONE;
                ek[2 * h + 1] = NONE;
            }
        }
    }

    // surplus cavity slots (a cavity with more tets than boundary faces) die, with a
    // forward pointer for the points located in them
    for (uint k = nb; k < nt; ++k) {
        node[4 * T[k]] = DEAD;
        neigh[4 * T[k]] = 4 * T[0];
        atomic_inc(&stats[7]);
    }

#undef SLOT
    state[p] = ST_INSERTED;
}

// end of the round: release every claim of every candidate
__kernel void d_reset(__global const uint* cavT, __global const uint* cavB, __global const uint* cnt,
                      __global uint* ownC, __global uint* ownS, __global const uint* stats, uint maxcand) {
    const uint ci = get_global_id(0);

    if (ci >= min(stats[1], maxcand)) {
        return;
    }

    const uint nt = cnt[2 * ci], nb = cnt[2 * ci + 1];
    __global const uint* T = cavT + (size_t)MAXC * ci;
    __global const uint* B = cavB + (size_t)MAXB * ci;

    for (uint i = 0; i < nt; ++i) {
        ownC[T[i]] = NONE;
    }

    for (uint i = 0; i < nb; ++i) {
        ownS[B[i] >> 2] = NONE;
    }
}
