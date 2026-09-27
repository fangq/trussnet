#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- command-line tests of the standalone binary (ctest: cli).
#
# usage: tests/run_cli_tests.sh path/to/trussnet[.exe] [work dir]
#
# The phantoms that are conforming at --dim 64 must stay exactly conforming (0 bad
# faces / 0 edges through label 0 / 0 spanning); the harder ones must only run.
# --gpu falls back to the CPU without an OpenCL device (the CI runners), so it is
# checked here as well. Exits non-zero on the first-class failures (all are run).

set -u
exe=${1:?usage: run_cli_tests.sh path/to/trussnet [work dir]}
wd=${2:-$(mktemp -d 2>/dev/null || echo ./tn_cli_tests)}
mkdir -p "$wd"
pass=0
fail=0

ok() {
    pass=$((pass + 1))
    printf '  PASS  %s\n' "$1"
}

bad() {
    fail=$((fail + 1))
    printf '  FAIL  %s: %s\n' "$1" "$2"
}

# run NAME ARGS...: the output in $out, the exit code in $rc
run() {
    local name=$1
    shift
    out=$("$exe" "$@" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
        bad "$name" "exit code $rc"
        printf '%s\n' "$out" | tail -5
        return 1
    fi
    return 0
}

same_file() {   # cmp is not on every Windows shell; cksum is POSIX
    if command -v cmp > /dev/null 2>&1; then
        cmp -s "$1" "$2"
    else
        [ "$(cksum < "$1")" = "$(cksum < "$2")" ]
    fi
}

conforming() {   # the [tess] line reports 0 / 0 / 0
    printf '%s\n' "$out" | grep -q "conformity: 0 bad faces, 0 edges through label 0, 0 spanning"
}

echo "trussnet CLI tests: $exe (work dir $wd)"

# ---- basics
if run help --help; then
    printf '%s\n' "$out" | grep -q "usage:" && ok help || bad help "no usage text"
fi

if run version --version; then
    printf '%s\n' "$out" | grep -Eq "^trussnet [0-9]+\.[0-9]+\.[0-9]+" && ok version || bad version "$out"
fi

if "$exe" --no-such-option > /dev/null 2>&1; then
    bad bad-option "an unknown option was accepted"
else
    ok bad-option
fi

if "$exe" -i "$wd/does-not-exist.nii.gz" > /dev/null 2>&1; then
    bad missing-input "a missing input file was accepted"
else
    ok missing-input
fi

# ---- 3-D phantoms: exactly conforming at --dim 64
for sh in sphere twoballs corrsphere gyroid torus ushape tjunction helix3 boxhemi sandwich shells hollow slab wedge \
          graysphere; do
    if run "$sh" --shape "$sh" --dim 64; then
        conforming && ok "$sh" || bad "$sh" "$(printf '%s\n' "$out" | grep 'tess\]' | sed 's/.*conformity: //')"
    fi
done

# the hard ones only have to run
for sh in holeysheet graygyroid; do
    run "$sh (runs)" --shape "$sh" --dim 64 && ok "$sh (runs)"
done

# ---- outputs: text / binary JMesh, and determinism
if run "output .jmsh" --shape twoballs --dim 48 -o "$wd/a.jmsh"; then
    if [ -s "$wd/a.jmsh" ] && grep -q '"MeshElem"' "$wd/a.jmsh"; then ok "output .jmsh"; else bad "output .jmsh" "no MeshElem"; fi
fi

if run "output .bmsh" --shape twoballs --dim 48 -o "$wd/a.bmsh"; then
    [ -s "$wd/a.bmsh" ] && ok "output .bmsh" || bad "output .bmsh" "empty file"
fi

if run determinism --shape twoballs --dim 48 -o "$wd/b.jmsh"; then
    same_file "$wd/a.jmsh" "$wd/b.jmsh" && ok determinism || bad determinism "two runs differ"
fi

# ---- options
if run lsize --shape shells --dim 48 --lsize 3:1.5; then
    conforming && ok lsize || bad lsize "not conforming"
fi

if run isize --shape shells --dim 48 --size 6 --isize 0:2,1:2:2; then
    conforming && ok isize || bad isize "not conforming"
fi

if run thin --shape shells --dim 48 --size 6 --isize 1:2:1.5 --grad 1 --thin 0.7; then
    if conforming && printf '%s\n' "$out" | grep -q "\[thin\]  [1-9][0-9]* of"; then ok thin; else bad thin "no nodes thinned or not conforming"; fi
fi

if "$exe" --shape sphere --dim 24 --tpm-thresh 2:1.5 > /dev/null 2>&1; then
    bad bad-tpm-thresh "a bad --tpm-thresh was accepted"
else
    ok bad-tpm-thresh
fi

if "$exe" --shape sphere --dim 24 --isize 1:2:3:4 > /dev/null 2>&1; then
    bad bad-isize "a bad --isize was accepted"
else
    ok bad-isize
fi

if run "gpu (or its CPU fallback)" --shape sphere --dim 48 --gpu; then
    conforming && ok "gpu (or its CPU fallback)" || bad "gpu (or its CPU fallback)" "not conforming"
fi

if run "voxel trapping" --shape sphere --dim 48 --trap voxel; then
    ok "voxel trapping"
fi

# the Jacobi relaxation (FIRE is the default): the conforming phantoms stay conforming
for sh in tjunction twoballs; do
    if run "jacobi $sh" --shape "$sh" --dim 64 --relax jacobi; then
        conforming && ok "jacobi $sh" || bad "jacobi $sh" "$(printf '%s\n' "$out" | grep 'tess\]' | sed 's/.*conformity: //')"
    fi
done

if "$exe" --shape sphere --dim 24 --relax verlet > /dev/null 2>&1; then
    bad bad-relax "a bad --relax was accepted"
else
    ok bad-relax
fi

# ---- --mode: the stages on their own
if run "mode mesh --faces" --shape twoballs --dim 48 --faces -o "$wd/mf.jmsh"; then
    if grep -q '"MeshElem"' "$wd/mf.jmsh" && grep -q '"MeshTri"' "$wd/mf.jmsh"; then ok "mode mesh --faces"; else bad "mode mesh --faces" "MeshElem / MeshTri missing"; fi
fi

if run "mode surface" --shape twoballs --dim 48 --mode surface -o "$wd/ms.jmsh"; then
    if grep -q '"MeshTri"' "$wd/ms.jmsh" && ! grep -q '"MeshElem"' "$wd/ms.jmsh" && "$exe" --mode check -i "$wd/ms.jmsh" > /dev/null 2>&1; then
        ok "mode surface"
    else
        bad "mode surface" "no surface, or it does not pass --mode check"
    fi
fi

if run "mode surface --exact-tess" --shape twoballs --dim 48 --mode surface --exact-tess -o "$wd/mse.jmsh"; then
    if grep -q '"MeshTri"' "$wd/mse.jmsh" && "$exe" --mode check -i "$wd/mse.jmsh" > /dev/null 2>&1; then
        ok "mode surface --exact-tess"
    else
        bad "mode surface --exact-tess" "no surface, or it does not pass --mode check"
    fi
fi

if run "mode points" --shape twoballs --dim 48 --mode points -o "$wd/mp.jmsh"; then
    if grep -q '"NodeType"' "$wd/mp.jmsh" && ! grep -q '"MeshElem"' "$wd/mp.jmsh"; then ok "mode points"; else bad "mode points" "no node attributes"; fi
fi

if run "mode check (tets)" --mode check -i "$wd/mf.jmsh"; then
    ok "mode check (tets)"
fi

# two interpenetrating tetrahedra's surfaces, and an open one: --mode check fails (exit 3)
printf 'OFF\n8 8 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n0.2 0.2 0.2\n1.2 0.2 0.2\n0.2 1.2 0.2\n0.2 0.2 1.2\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n3 4 6 5\n3 4 5 7\n3 4 7 6\n3 5 6 7\n' > "$wd/cross.off"
printf 'OFF\n4 3 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 2 1\n3 0 1 3\n3 0 3 2\n' > "$wd/open.off"

for f in cross open; do
    "$exe" --mode check -i "$wd/$f.off" > /dev/null 2>&1
    rc=$?

    if [ $rc -eq 3 ]; then ok "check finds $f"; else bad "check finds $f" "exit code $rc (want 3)"; fi
done

# optimize: an unoptimised mesh, then the optimiser alone (it must stay valid)
if run "mode optimize" --shape tjunction --dim 48 --opt 0 -o "$wd/mo0.jmsh"; then
    if "$exe" --mode optimize -i "$wd/mo0.jmsh" -o "$wd/mo1.jmsh" > "$wd/mo.log" 2>&1 && grep -q '"MeshElem"' "$wd/mo1.jmsh" &&
            "$exe" --mode check -i "$wd/mo1.jmsh" > /dev/null 2>&1; then
        ok "mode optimize"
    else
        bad "mode optimize" "$(tail -3 "$wd/mo.log")"
    fi
fi

# tessellate: a point cloud (the corners and centre of a cube) -> its Delaunay tets
printf '0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n0.5 0.5 0.5\n' > "$wd/cube.xyz"

if run "mode tessellate" --mode tessellate -i "$wd/cube.xyz" -o "$wd/mt.jmsh"; then
    grep -q '"MeshElem"' "$wd/mt.jmsh" && ok "mode tessellate" || bad "mode tessellate" "no MeshElem"
fi

# the relaxed nodes, then the tessellation against the same volume: the one-shot mesh
if run "points -> tessellate" --shape twoballs --dim 48 --mode points -o "$wd/rt_p.jmsh"; then
    if "$exe" --shape twoballs --dim 48 --mode tessellate -i "$wd/rt_p.jmsh" -o "$wd/rt_t.jmsh" > /dev/null 2>&1 &&
            "$exe" --shape twoballs --dim 48 -o "$wd/rt_m.jmsh" > /dev/null 2>&1 && same_file "$wd/rt_t.jmsh" "$wd/rt_m.jmsh"; then
        ok "points -> tessellate"
    else
        bad "points -> tessellate" "differs from the one-shot mesh"
    fi
fi

# repair: the interpenetrating tetrahedra above -> one clean, closed surface
if run "mode repair" --mode repair -i "$wd/cross.off" --raster-voxel 0.03 --size 0.1 -o "$wd/cross_r.jmsh"; then
    if "$exe" --mode check -i "$wd/cross_r.jmsh" > /dev/null 2>&1; then ok "mode repair"; else bad "mode repair" "the repaired surface fails --mode check"; fi
fi

# remesh: a tetrahedron nested in another (no labels) -> two labelled regions
printf 'OFF\n8 8 0\n0 0 0\n2 0 0\n0 2 0\n0 0 2\n0.3 0.3 0.3\n0.8 0.3 0.3\n0.3 0.8 0.3\n0.3 0.3 0.8\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n3 4 6 5\n3 4 5 7\n3 4 7 6\n3 5 6 7\n' > "$wd/nested.off"

if run "mode remesh" --mode remesh -i "$wd/nested.off" --raster-voxel 0.03 --size 0.1 -o "$wd/nested_m.jmsh"; then
    if printf '%s\n' "$out" | grep -q "2 region(s)" && "$exe" --mode check -i "$wd/nested_m.jmsh" > /dev/null 2>&1; then
        ok "mode remesh"
    else
        bad "mode remesh" "not two regions, or the mesh fails --mode check"
    fi
fi

# cdt: the region surfaces of a mesh -> tets keeping them exactly; a crossing input is refused
if run "mode cdt" --mode cdt -i "$wd/ms.jmsh" -o "$wd/cdt.jmsh"; then
    if grep -q '"MeshElem"' "$wd/cdt.jmsh" && "$exe" --mode check -i "$wd/cdt.jmsh" > /dev/null 2>&1; then
        ok "mode cdt"
    else
        bad "mode cdt" "no tets, or they fail --mode check"
    fi
fi

if "$exe" --mode cdt -i "$wd/cross.off" > /dev/null 2>&1; then
    bad "cdt refuses crossings" "a self-intersecting surface was accepted"
else
    ok "cdt refuses crossings"
fi

if "$exe" --shape sphere --dim 24 --mode nosuchmode > /dev/null 2>&1; then
    bad bad-mode "a bad --mode was accepted"
else
    ok bad-mode
fi

# ---- 2-D: a single slice -> triangles
for sh in disk2d gray2d; do
    if run "$sh" --shape "$sh" --dim 128 --size 3 -o "$wd/$sh.jmsh"; then
        if printf '%s\n' "$out" | grep -q "conformity: 0 bad edges, 0 spanning" && grep -q '"MeshTri"' "$wd/$sh.jmsh"; then
            ok "$sh"
        else
            bad "$sh" "$(printf '%s\n' "$out" | grep '\[2d\]')"
        fi
    fi
done

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
