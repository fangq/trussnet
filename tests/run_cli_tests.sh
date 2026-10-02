#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# v2mesh -- command-line tests of the standalone binary (ctest: cli).
#
# usage: tests/run_cli_tests.sh path/to/v2mesh[.exe] [work dir]
#
# The phantoms that are conforming at --dim 64 must stay exactly conforming (0 bad
# faces / 0 edges through label 0 / 0 spanning); the harder ones must only run.
# --gpu falls back to the CPU without an OpenCL device (the CI runners), so it is
# checked here as well. Exits non-zero on the first-class failures (all are run).

set -u
exe=${1:?usage: run_cli_tests.sh path/to/v2mesh [work dir]}
wd=${2:-$(mktemp -d 2>/dev/null || echo ./v2m_cli_tests)}
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

echo "v2mesh CLI tests: $exe (work dir $wd)"

# ---- basics
if run help --help; then
    printf '%s\n' "$out" | grep -q "usage:" && ok help || bad help "no usage text"
fi

if run version --version; then
    printf '%s\n' "$out" | grep -Eq "^v2mesh [0-9]+\.[0-9]+\.[0-9]+" && ok version || bad version "$out"
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

# a tet mesh as the surfaces: its region surfaces
if run "cdt of a tet mesh" --mode cdt -i "$wd/mo1.jmsh" -o "$wd/cdt2.jmsh" && "$exe" --mode check -i "$wd/cdt2.jmsh" > /dev/null 2>&1; then
    ok "cdt of a tet mesh"
elif [ -f "$wd/cdt2.jmsh" ]; then
    bad "cdt of a tet mesh" "the tets fail --mode check"
fi

# TetGen PLCs (.poly / .smesh): polygon facets (one with a hole), regions, a volume hole
data=$(cd "$(dirname "$0")" && pwd)/data
for c in "frame.poly|1:840" "split.poly|5:500 7:500" "cavity.poly|1:992" "splits.smesh|3:500 5:500"; do
    f=${c%%|*}; want=${c#*|}
    if run "plc $f" --mode cdt -i "$data/$f" -o "$wd/plc.jmsh"; then
        got=$("$exe" --mode check -i "$wd/plc.jmsh" 2>&1 | sed -n 's/.*volume per label: //p' | tail -1)
        if [ "$got" = "$want" ] && "$exe" --mode check -i "$wd/plc.jmsh" > /dev/null 2>&1; then
            ok "plc $f ($got)"
        else
            bad "plc $f" "volume per label: $got (want $want), or it fails --mode check"
        fi
    fi
done

# a TetGen example (a PM DC motor): its points in the .node its comment names,
# polygon lists wrapped over lines, 4 region seeds; closed, meshed by cdt
if run "plc pdmc.poly" --mode cdt --opt 0 -i "$data/pdmc.poly" -o "$wd/plc.jmsh" &&
        printf '%s\n' "$out" | grep -q "972 points, 502 facets" && "$exe" --mode check -i "$wd/plc.jmsh" > /dev/null 2>&1; then
    ok "plc pdmc.poly ($("$exe" --mode check -i "$wd/plc.jmsh" 2>&1 | sed -n 's/.*volume per label: //p' | tail -1))"
elif [ -f "$wd/plc.jmsh" ]; then
    bad "plc pdmc.poly" "not 972 points / 502 facets, or the tets fail --mode check"
fi
# a volume bound (--maxvol, TetGen -a): no tet above it, every region's volume (so
# its surface) as without it -- the large tets on the surface triangles bisected
if run "cdt --maxvol" --mode cdt -i "$data/pdmc.poly" --size 4 --maxvol 5 -o "$wd/plcv.jmsh"; then
    ck=$("$exe" --mode check -i "$wd/plcv.jmsh" 2>&1)
    big=$(printf '%s\n' "$ck" | sed -n 's/.*largest tet \([0-9.e+]*\).*/\1/p' | tail -1)
    lv=$(printf '%s\n' "$ck" | sed -n 's/.*volume per label: //p' | tail -1)
    lv0=$("$exe" --mode check -i "$wd/plc.jmsh" 2>&1 | sed -n 's/.*volume per label: //p' | tail -1)
    if printf '%s\n' "$ck" | grep -q "\[check\] OK" && awk -v v="$big" 'BEGIN { exit !(v != "" && v <= 5.0001) }' && [ "$lv" = "$lv0" ]; then
        ok "cdt --maxvol (largest tet $big)"
    else
        bad "cdt --maxvol" "largest tet '$big', volumes '$lv' (want '$lv0'), or the check fails"
    fi
fi

# STEP (CAD B-reps, written by Open CASCADE): watertight surfaces, the volume within
# 1 % of the exact one (the tessellation's chords): a pole-only sphere, a torus
# (both directions periodic), a cone's apex, a band (a hole through a box), a
# rational B-spline loft, and a sphere face with a hole (its complement)
for c in "sphere|523.59878" "torus|473.74101" "cone|134.04129" "boxhole|717.25666" "loft|238.70217" "spherebox|288.46231"; do
    f=${c%%|*}; want=${c#*|}
    if run "step $f" --mode cdt -i "$data/$f.step" -o "$wd/step.jmsh"; then
        got=$("$exe" --mode check -i "$wd/step.jmsh" 2>&1 | sed -n 's/.*volume per label: 1://p' | tail -1)
        if printf '%s\n' "$out" | grep -q "(0 failed)" && "$exe" --mode check -i "$wd/step.jmsh" > /dev/null 2>&1 &&
                awk -v g="$got" -v w="$want" 'BEGIN { exit !(g > 0.99 * w && g < 1.01 * w) }'; then
            ok "step $f ($got, exact $want)"
        else
            bad "step $f" "volume $got (exact $want), a face failed, or it fails --mode check"
        fi
    fi
done

# shape constructs (JSON): exact signed distance functions, sharp features pinned
printf '{"Shapes":[{"Box":{"O":[20,20,20],"Size":[60,50,40],"Tag":1}}]}\n' > "$wd/sbox.json"
printf '{"Shapes":[{"Grid":{"Size":[40,40,40],"Tag":0}},{"Sphere":{"O":[20,20,20],"R":14,"Tag":1}},{"Sphere":{"O":[20,20,20],"R":7,"Tag":2}}]}\n' > "$wd/ssph.json"
printf '{"ShapeBox3(outer)":{"O":[0,0,0],"P":[40,30,30],"Tag":1},"ShapeSphere(hole)":{"O":[20,15,30],"R":9},"CSGObject(dented)":[{"CSGSubtract":["outer","hole"]},{"Tag":1}],"ShapeCylinder":{"O":[8,15,4],"P":[8,15,26],"R":5,"Tag":2},"ShapeTorus":{"O":[30,15,12],"R":6,"Rtube":2.5,"N":[0,0,1],"Tag":3}}\n' > "$wd/scsg.json"
printf '{"Shapes":[{"Box":{"O":[0,0,0],"Size":[20,20,20],"Tag":1}},{"Sphere":{"O":[20,10,10],"R":6,"Tag":2}}]}\n' > "$wd/sclip.json"
labvols() { "$exe" --mode check -i "$1" 2>&1 | sed -n 's/.*volume per label: //p' | tail -1; }
if run "shapes: box" -i "$wd/sbox.json" --size 6 -v -o "$wd/sbox.jmsh" && conforming; then
    v=$(labvols "$wd/sbox.jmsh" | sed -n 's/^1:\([0-9.e+]*\).*/\1/p')
    if awk -v v="$v" 'BEGIN { exit !(v > 119999 && v < 120001) }' &&
            printf '%s\n' "$out" | awk '/^\[sdf\]/ { split($0, a, "max "); split(a[2], b, " "); exit !(b[1] < 1e-3) }'; then
        ok "shapes: box exact ($v), surface nodes on the shapes"
    else
        bad "shapes: box" "volume $v (120000), or [sdf] off: $(printf '%s\n' "$out" | grep '^\[sdf\]')"
    fi
fi
# straight creases (exact volumes): a box through the domain's wall, on it, and cut
# by another box (the crossing curves pinned, the pinned pairs kept joined)
# (and a notch inside one object's CSG: a box less a box, its corners exact)
printf '{"ShapeBox3(a)":{"O":[0,0,0],"P":[20,20,20]},"ShapeBox3(b)":{"O":[10,10,10],"P":[30,30,30]},"CSGObject":[{"CSGSubtract":["a","b"]},{"Tag":1}]}\n' > "$wd/snotch.json"
if run "shapes: crease notch" -i "$wd/snotch.json" --size 3 -o "$wd/snotch.jmsh" && conforming; then
    v=$(labvols "$wd/snotch.jmsh" | sed -n 's/^1:\([0-9.e+]*\).*/\1/p')
    if awk -v v="$v" 'BEGIN { exit !(v > 6999 && v < 7001) }'; then ok "shapes: crease notch ($v)"; else bad "shapes: crease notch" "volume $v (7000)"; fi
fi
# (and a knife edge: a box less a larger sphere, its holes' rims 31 degrees -- no gap
# closing inside one object's CSG, the rims pinned and refined: within 1.5 % of exact)
printf '{"Shapes":[{"CSGObject":[{"CSGSubtract":[{"Grid":{"Size":[60,60,60]}},{"Sphere":{"O":[30,30,30],"R":35}}]},{"Tag":1}]}]}\n' > "$wd/sknife.json"
# (its tets' edges may dip into the sphere at the rims -- chords of a thin wedge -- so the
# check here is --mode check and the volume, not a zero conformity count)
if run "shapes: knife edge" -i "$wd/sknife.json" --size 3 -o "$wd/sknife.jmsh" &&
        "$exe" --mode check -i "$wd/sknife.jmsh" > /dev/null 2>&1; then
    v=$(labvols "$wd/sknife.jmsh" | sed -n 's/^1:\([0-9.e+]*\).*/\1/p')
    if awk -v v="$v" 'BEGIN { exit !(v > 0.985 * 52113.6 && v < 1.005 * 52113.6) }'; then ok "shapes: knife edge ($v, exact 52113.6)"
    else bad "shapes: knife edge" "volume $v (52113.6)"; fi
fi
# exact-surface CSG (--mode cdt of shapes): the constructs' own surfaces, crossings
# traced exactly -- the knife rim, a drilled hole, coincident faces (a layer, a corner
# block, a cap on the domain's face), a torus cut at its equator, a triple junction;
# every label's volume exact (planes) or within the chord tolerance (curved)
csgcase() {   # $1 name, $2 JSON, $3 "label:volume:relative tolerance ..."
    printf '%s\n' "$2" > "$wd/csg.json"
    if run "csg: $1" --mode cdt -i "$wd/csg.json" --size 3 -o "$wd/csg.jmsh" &&
            "$exe" --mode check -i "$wd/csg.jmsh" > /dev/null 2>&1; then
        got=$(labvols "$wd/csg.jmsh")
        for w in $3; do
            l=${w%%:*}; r=${w#*:}; want=${r%%:*}; rt=${r#*:}
            v=$(printf '%s\n' "$got" | tr ' ' '\n' | sed -n "s/^$l://p")
            if ! awk -v v="$v" -v w="$want" -v t="$rt" 'BEGIN { d = v - w; if (d < 0) d = -d; exit !(v != "" && d <= t * w) }'; then
                bad "csg: $1" "label $l volume '$v' (exact $want)"
                return
            fi
        done
        ok "csg: $1 ($got)"
    elif [ $rc -eq 0 ]; then
        bad "csg: $1" "the tets fail --mode check"
    fi
}
g='{"Grid":{"Size":[60,60,60],"Tag":1}}'
csgcase "knife rim" '{"Shapes":[{"CSGObject":[{"CSGSubtract":[{"Grid":{"Size":[60,60,60]}},{"Sphere":{"O":[30,30,30],"R":35}}]},{"Tag":1}]}]}' \
    "1:52113.6:0.008"
csgcase "drilled hole" '{"Shapes":[{"CSGObject":[{"CSGSubtract":[{"Grid":{"Size":[60,60,60]}},{"Cylinder":{"C0":[30,30,-5],"C1":[30,30,65],"R":10}}]},{"Tag":1}]}]}' \
    "1:197150.4:0.001"
csgcase "layer" "{\"Shapes\":[$g,{\"Box\":{\"O\":[0,0,20],\"Size\":[60,60,10],\"Tag\":2}}]}" "1:180000:1e-5 2:36000:1e-5"
csgcase "corner block" "{\"Shapes\":[$g,{\"Box\":{\"O\":[0,0,0],\"Size\":[30,30,30],\"Tag\":2}}]}" "1:189000:1e-5 2:27000:1e-5"
csgcase "cap on a face" "{\"Shapes\":[$g,{\"Cylinder\":{\"C0\":[30,30,0],\"C1\":[30,30,20],\"R\":10,\"Tag\":2}},{\"Sphere\":{\"O\":[30,30,0],\"R\":6,\"Tag\":3}}]}" \
    "2:5830.8:0.008 3:452.4:0.02"
csgcase "torus cut at its equator" "{\"Shapes\":[$g,{\"CSGObject\":[{\"CSGSubtract\":[{\"ShapeTorus\":{\"O\":[30,30,30],\"N\":[0,0,1],\"R\":15,\"Rtube\":5}},{\"Box\":{\"O\":[0,0,30],\"Size\":[60,60,30]}}]},{\"Tag\":2}]}]}" \
    "2:3701.1:0.012"
csgcase "triple junction" "{\"Shapes\":[$g,{\"CSGObject\":[{\"CSGUnion\":[{\"Sphere\":{\"O\":[25,30,30],\"R\":12}},{\"Sphere\":{\"O\":[35,31,29],\"R\":11}},{\"Sphere\":{\"O\":[29,38,32],\"R\":10}}]},{\"Tag\":2}]}]}" \
    "2:11848.7:0.012"
for c in "wall|40,12,10|2:14000" "onwall|0,12,10|2:21000" "cut|10,10,10|3:11000" "cutcells|10,10,10|4:4199"; do
    nm=${c%%|*}; rest=${c#*|}; o=${rest%%|*}; want=${rest#*|}
    if [ "${nm#cut}" != "$nm" ]; then
        printf '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,50,50]}},{"Box":{"Tag":2,"O":[%s],"Size":[25,25,25]}},{"Box":{"Tag":3,"O":[22,18,16],"Size":[25,20,22]}}]}\n' "$o" > "$wd/scr.json"
    else
        printf '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,50,50]}},{"Box":{"Tag":2,"O":[%s],"Size":[30,25,28]}}]}\n' "$o" > "$wd/scr.json"
    fi
    extra=""; [ "$nm" = cutcells ] && extra="--overlap cells"
    # shellcheck disable=SC2086
    if run "shapes: crease $nm" -i "$wd/scr.json" --size 3 $extra -o "$wd/scr.jmsh" && conforming; then
        v=$(labvols "$wd/scr.jmsh" | tr ' ' '\n' | sed -n "s/^${want%%:*}://p")
        if awk -v v="$v" -v w="${want#*:}" 'BEGIN { exit !(v > w - 1 && v < w + 1) }'; then
            ok "shapes: crease $nm (label ${want%%:*}: $v)"
        else
            bad "shapes: crease $nm" "label ${want%%:*}: $v (${want#*:})"
        fi
    fi
done
if run "shapes: spheres" -i "$wd/ssph.json" --size 2 -o "$wd/ssph.jmsh" && conforming; then
    # MCX order: the inner sphere overwrites the outer; within 3% of the analytic volumes
    if labvols "$wd/ssph.jmsh" | awk '{ split($1, a, ":"); split($2, b, ":"); o = 4/3*3.14159265*(14^3-7^3); i = 4/3*3.14159265*7^3;
            exit !(a[2] > 0.97*o && a[2] < 1.01*o && b[2] > 0.97*i && b[2] < 1.01*i) }'; then
        ok "shapes: nested spheres"
    else
        bad "shapes: nested spheres" "$(labvols "$wd/ssph.jmsh")"
    fi
fi
# (tr: BSD wc pads its count with spaces)
if run "shapes: JMesh CSG" -i "$wd/scsg.json" --size 2 -o "$wd/scsg.jmsh" && conforming &&
        [ "$(labvols "$wd/scsg.jmsh" | wc -w | tr -d " ")" = 3 ]; then
    ok "shapes: JMesh CSG"
else
    bad "shapes: JMesh CSG" "$(labvols "$wd/scsg.jmsh"); $(printf '%s\n' "$out" | grep -oE 'conformity: [0-9]+ bad faces, [0-9]+ edges through label 0, [0-9]+ spanning')"
fi
# overlapping objects (--overlap): two spheres that overlap, inside a box
printf '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,50,50]}},{"Sphere":{"Tag":2,"O":[24,25,25],"R":12}},{"Sphere":{"Tag":3,"O":[36,25,25],"R":9}}]}\n' > "$wd/sov.json"
v2over=""; v2under=""
for c in "overwrite|3" "nest|3" "max|3" "min|3" "order:2,3|3" "split|3" "union|2" "cells|4"; do
    r=${c%|*}; n=${c#*|}
    if run "shapes: overlap $r" -i "$wd/sov.json" --size 3 --overlap "$r" -o "$wd/sov.jmsh" && conforming &&
            [ "$(labvols "$wd/sov.jmsh" | wc -w | tr -d " ")" = "$n" ]; then
        ok "shapes: overlap $r"
        v2=$(labvols "$wd/sov.jmsh" | tr ' ' '\n' | sed -n 's/^2://p')
        [ "$r" = overwrite ] && v2under=$v2   # (sphere 2 loses the overlap)
        [ "$r" = min ] && v2over=$v2          # (sphere 2 wins it)
        [ "$r" = split ] && v2split=$v2
    else
        bad "shapes: overlap $r" "$(labvols "$wd/sov.jmsh"); $(printf '%s\n' "$out" | grep -oE 'conformity: [0-9]+ bad faces, [0-9]+ edges through label 0, [0-9]+ spanning')"
    fi
done
# split: sphere 2's share of the overlap between losing it and winning it
if [ -n "$v2under" ] && [ -n "$v2over" ] && [ -n "${v2split:-}" ] &&
        awk -v u="$v2under" -v o="$v2over" -v s="$v2split" 'BEGIN { exit !(s > u && s < o) }'; then
    ok "shapes: overlap split shares"
else
    bad "shapes: overlap split shares" "sphere 2: overwrite $v2under, split ${v2split:-?}, min $v2over"
fi
# cells: a cylinder through a sphere -- two crossing curves, four regions round each
printf '{"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},{"Sphere":{"Tag":2,"O":[30,30,30],"R":12}},{"Cylinder":{"Tag":3,"C0":[30,30,5],"C1":[30,30,55],"R":5}}]}\n' > "$wd/sovc.json"
if run "shapes: overlap cells (cylinder)" -i "$wd/sovc.json" --size 3 --overlap cells -o "$wd/sovc.jmsh" && conforming &&
        [ "$(labvols "$wd/sovc.jmsh" | wc -w | tr -d " ")" = 4 ]; then
    ok "shapes: overlap cells (cylinder)"
else
    bad "shapes: overlap cells (cylinder)" "$(labvols "$wd/sovc.jmsh"); $(printf '%s\n' "$out" | grep -oE 'conformity: [0-9]+ bad faces, [0-9]+ edges through label 0, [0-9]+ spanning')"
fi
if run "shapes: clip" --shape "$wd/sclip.json" --size 2 --shape-clip 0 -o "$wd/sclip0.jmsh" &&
        run "shapes: clip" --shape "$wd/sclip.json" --size 2 -o "$wd/sclip1.jmsh"; then
    # (not cut: the sphere pokes out of the box -- larger than when cut to it)
    v0=$(labvols "$wd/sclip0.jmsh" | sed -n 's/.*2:\([0-9.e+]*\).*/\1/p'); v1=$(labvols "$wd/sclip1.jmsh" | sed -n 's/.*2:\([0-9.e+]*\).*/\1/p')
    if awk -v a="$v0" -v b="$v1" 'BEGIN { exit !(a > 1.6 * b) }'; then
        ok "shapes: clip ($v1 cut, $v0 not)"
    else
        bad "shapes: clip" "sphere $v1 cut, $v0 not"
    fi
fi
# tangent shapes: the box's bottom touching the inner sphere (zero thickness
# between them) -- the gap closes, conforming
printf '{"Shapes":[{"Grid":{"Tag":0,"Size":[100,100,100]}},{"Sphere":{"O":[50,50,50],"R":35,"Tag":2}},{"Sphere":{"O":[50,50,50],"R":20,"Tag":3}},{"Cylinder":{"C0":[0,50,50],"C1":[100,50,50],"R":10,"Tag":4}},{"Box":{"O":[35,35,70],"Size":[30,30,20],"Tag":5}}]}\n' > "$wd/stan.json"
printf '{"Shapes":[{"Grid":{"Size":[40,40,40],"Tag":0}},{"Sphere":{"O":[20,20,20],"R":14,"Tag":1}},{"Sphere":{"O":[20,20,18],"R":8,"Tag":2}},{"Box":{"O":[14,14,26],"Size":[12,12,8],"Tag":3}}]}\n' > "$wd/stan2.json"
# (+ an edge piercing a surface at a coarse size; a triple line crowded at a fine one)
printf '{"Shapes":[{"Grid":{"Tag":0,"Size":[100,100,100]}},{"Sphere":{"O":[50,50,50],"R":35,"Tag":2}},{"Sphere":{"O":[50,50,50],"R":20,"Tag":3}},{"Cylinder":{"C0":[0,50,50],"C1":[100,50,50],"R":10,"Tag":4}},{"Box":{"O":[35,35,72],"Size":[30,30,20],"Tag":5}}]}\n' > "$wd/sblc.json"
for c in "stan|4" "stan2|2" "stan2|4" "sblc|2"; do
    f=${c%%|*}; h=${c#*|}
    if run "shapes: tangent $f size $h" -i "$wd/$f.json" --size "$h" -o "$wd/$f.jmsh" && conforming; then
        ok "shapes: tangent $f size $h"
    else
        bad "shapes: tangent $f size $h" "$(printf '%s\n' "$out" | grep -oE 'conformity: [0-9]+ bad faces, [0-9]+ edges through label 0, [0-9]+ spanning')"
    fi
done
printf '{"Shapes":[{"Blob":{"O":[0,0,0]}}]}\n' > "$wd/sbad.json"
if "$exe" -i "$wd/sbad.json" > /dev/null 2>&1; then
    bad "shapes: bad" "an unknown construct was accepted"
else
    ok "shapes: bad"
fi

# --manifold: two cubes touching only along an edge (a voxel checkerboard)
# pinch the surface there; opened, no edge is on more than two faces. --nest
# 1,2: the same cubes (label 2) inside label 1, the inner label joined
mkdiag() {   # $1: the label round the cubes (0: none)
    awk -v bg="$1" 'BEGIN { n = 16; printf "{\"NIFTIHeader\":{\"Dim\":[%d,%d,%d],\"VoxelSize\":[1,1,1],\"Affine\":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]},\"NIFTIData\":{\"_ArrayType_\":\"uint8\",\"_ArraySize_\":[%d,%d,%d],\"_ArrayData_\":[", n, n, n, n, n, n;
        c = 0; for (i = 0; i < n; i++) for (j = 0; j < n; j++) for (k = 0; k < n; k++) {
            cube = ((i >= 3 && i < 8 && j >= 3 && j < 8) || (i >= 8 && i < 13 && j >= 8 && j < 13)) && k >= 3 && k < 13;
            inside = bg > 0 && i >= 1 && i < 15 && j >= 1 && j < 15 && k >= 1 && k < 15;
            printf "%s%d", c++ ? "," : "", cube ? (bg > 0 ? 2 : 1) : (inside ? bg : 0) } printf "]}}\n" }'
}
mkdiag 0 > "$wd/diag.jnii"
mkdiag 1 > "$wd/diag2.jnii"
# volume-preserving surface smoothing (--surf-smooth, iso2mesh smoothsurf's HC): the
# region surfaces smoother, the mesh still valid, the volumes within 2 % of unsmoothed
# (iso2mesh's alpha = beta = 0.5; the coarse balls are the worst case: -1.4 %)
if run "surf-smooth" --shape twoballs --dim 48 -o "$wd/ss0.jmsh" && run "surf-smooth" --shape twoballs --dim 48 --surf-smooth 10 -o "$wd/ss1.jmsh"; then
    c0=$("$exe" --mode check -i "$wd/ss0.jmsh" 2>&1); c1=$("$exe" --mode check -i "$wd/ss1.jmsh" 2>&1)
    v0=$(printf '%s\n' "$c0" | sed -n 's/.*volume per label: //p' | tail -1); v1=$(printf '%s\n' "$c1" | sed -n 's/.*volume per label: //p' | tail -1)
    if printf '%s\n' "$c1" | grep -q "\[check\] OK" && awk -v a="$v0" -v b="$v1" 'BEGIN { n = split(a, x, " "); split(b, y, " ");
            for (i = 1; i <= n; i++) { split(x[i], p, ":"); split(y[i], q, ":"); if (p[2] <= 0 || (q[2] - p[2]) / p[2] > 0.02 || (p[2] - q[2]) / p[2] > 0.02) exit 1 } }'; then
        ok "surf-smooth ($v0 -> $v1)"
    else
        bad "surf-smooth" "volumes '$v0' -> '$v1', or the check fails"
    fi
fi
# a JNIfTI header without a proper Affine (savejnifti of an Analyze 7.5 file writes
# zeros, Orientation r/a/s): the voxel size and the Orientation letters instead --
# here p/s/l, so the third voxel axis (1.5 mm) is the canonical x
for c in "zero|\"Affine\":[[0,0,0,0],[0,0,0,0],[0,0,0,0]],\"Orientation\":{\"x\":\"r\",\"y\":\"a\",\"z\":\"s\"}|1 x 1 x 1.5" \
         "none|\"Orientation\":{\"x\":\"p\",\"y\":\"s\",\"z\":\"l\"}|1.5 x 1 x 1"; do
    nm=${c%%|*}; r=${c#*|}; hdr=${r%%|*}; want=${r#*|}
    mkdiag 0 | sed "s/\"VoxelSize\":\[1,1,1\],\"Affine\":\[\[1,0,0,0\],\[0,1,0,0\],\[0,0,1,0\]\]/\"VoxelSize\":[1,1,1.5],$hdr/" > "$wd/hdr_$nm.jnii"
    if run "jnifti affine: $nm" -i "$wd/hdr_$nm.jnii" --size 2 -o "$wd/hdr_$nm.jmsh"; then
        if printf '%s\n' "$out" | grep -q "voxels ($want mm)"; then ok "jnifti affine: $nm ($want mm)"
        else bad "jnifti affine: $nm" "$(printf '%s\n' "$out" | grep -o 'voxels ([^)]*)')"; fi
    fi
done
junctions() { "$exe" --mode check -i "$1" 2>&1 | sed -n 's/.* \([0-9]*\) junction edges.*/\1/p'; }
if run "manifold (pinched)" -i "$wd/diag.jnii" --size 1.5 -o "$wd/diag_p.jmsh" && run "manifold" -i "$wd/diag.jnii" --size 1.5 --manifold -o "$wd/diag_m.jmsh"; then
    jp=$(junctions "$wd/diag_p.jmsh"); jm=$(junctions "$wd/diag_m.jmsh")
    if [ -n "$jp" ] && [ "$jp" -gt 0 ] && [ "$jm" = 0 ]; then
        ok "manifold: $jp pinched edges opened"
    else
        bad "manifold" "pinched edges $jp -> $jm"
    fi
fi
if run "manifold --nest" -i "$wd/diag2.jnii" --size 1.5 --nest 1,2 -o "$wd/diag_n.jmsh"; then
    jn=$(junctions "$wd/diag_n.jmsh")
    if [ "$jn" = 0 ]; then
        ok "manifold --nest"
    else
        bad "manifold --nest" "$jn pinched edges left"
    fi
fi

if "$exe" --shape sphere --dim 24 --nest > /dev/null 2>&1; then
    bad bad-nest "an empty --nest was accepted"
else
    ok bad-nest
fi

# the fast surface path (surface nodes only): a single region's surface is a
# manifold -- no edge on 4+ faces (a surface pinched by fins / pockets)
for sh in sphere ushape; do
    if run "surface manifold $sh" --shape $sh --dim 40 --mode surface -o "$wd/sm_$sh.jmsh" &&
            "$exe" --mode check -i "$wd/sm_$sh.jmsh" 2>&1 | grep -q ' 0 junction edges'; then
        ok "surface manifold $sh"
    elif [ -f "$wd/sm_$sh.jmsh" ]; then
        bad "surface manifold $sh" "$("$exe" --mode check -i "$wd/sm_$sh.jmsh" 2>&1 | grep -oE '[0-9]+ junction edges')"
    fi
done

# mesh-only -q: circumcentres inside the regions -- better tets, the same surfaces and volumes
jlp5() { printf '%s\n' "$1" | sed -n 's/.*\[check\] tets:.* p5 \([0-9.]*\) .*/\1/p' | tail -1; }
totvol() { printf '%s\n' "$1" | sed -n 's/.*\[check\] tets:.*; volume \([0-9.e+]*\).*/\1/p' | tail -1; }
q0=$("$exe" --mode cdt --cdt-fill 0 -q 0 -i "$wd/ms.jmsh" 2>&1)
q1=$("$exe" --mode cdt --cdt-fill 0 -q 1.4 -i "$wd/ms.jmsh" 2>&1)
if [ -n "$(jlp5 "$q1")" ] && awk -v a="$(jlp5 "$q0")" -v b="$(jlp5 "$q1")" 'BEGIN { exit !(b > a) }' &&
        [ -n "$(totvol "$q0")" ] && [ "$(totvol "$q0")" = "$(totvol "$q1")" ] &&
        printf '%s\n' "$q1" | grep -q '0 self-intersections'; then
    ok "cdt -q refinement"
else
    bad "cdt -q refinement" "Joe-Liu p5 $(jlp5 "$q0") -> $(jlp5 "$q1")"
fi

# surfaces without inner / outer labels: exact regions from the cells
# (two tetrahedra sharing a face: 1/3 and 1/6; nested ones: 1.3125 and 1/48)
printf 'OFF\n5 7 0\n0 0 0\n1 0 0\n0 1 0\n0.2 0.2 1\n0.2 0.2 -2\n3 0 1 3\n3 1 2 3\n3 2 0 3\n3 1 0 4\n3 2 1 4\n3 0 2 4\n3 0 1 2\n' > "$wd/bip.off"
printf '{"MeshNode":[[0,0,0],[1,0,0],[0,1,0],[0.2,0.2,1],[0.2,0.2,-2]],"MeshTri":[[1,2,4,2],[2,3,4,2],[3,1,4,2],[2,1,5,1],[3,2,5,1],[1,3,5,1],[1,2,3,2]]}\n' > "$wd/bip1.jmsh"
tn='[[0,0,0],[2,0,0],[0,2,0],[0,0,2],[0.3,0.3,0.3],[0.8,0.3,0.3],[0.3,0.8,0.3],[0.3,0.3,0.8]]'
printf '{"MeshNode":%s,"MeshTri":[[1,3,2,5],[1,2,4,5],[1,4,3,5],[2,3,4,5],[5,7,6,7],[5,6,8,7],[5,8,7,7],[6,7,8,7]]}\n' "$tn" > "$wd/nest57.jmsh"
printf '{"MeshNode":%s,"MeshTri":[[1,3,2,5],[1,2,4,5],[1,4,3,5],[2,3,4,5],[5,7,6,5],[5,6,8,5],[5,8,7,5],[6,7,8,5]]}\n' "$tn" > "$wd/hollow55.jmsh"
cdtvol() { "$exe" --mode cdt --opt 0 -i "$1" 2>&1 | sed -n 's/.*volume per label: //p' | tail -1; }
for c in "bip.off|1:0.333333 2:0.166667|unlabelled, touching" "bip1.jmsh|1:0.333333 2:0.166667|a label per face, shared face once" \
         "nest57.jmsh|5:1.3125 7:0.0208333|a label per face, nested" "hollow55.jmsh|5:1.3125|a label per face, hollow"; do
    f=${c%%|*}; r=${c#*|}; want=${r%%|*}; what=${r#*|}
    got=$(cdtvol "$wd/$f")
    if [ "$got" = "$want" ]; then
        ok "cdt regions: $what"
    else
        bad "cdt regions: $what" "volumes '$got', want '$want'"
    fi
done

# crossing surfaces: the overlap rule decides the regions
for c in "nest|2" "union|1" "cells|3"; do
    r=${c%%|*}; want=${c#*|}
    got=$("$exe" --mode remesh --overlap "$r" -i "$wd/cross.off" --raster-voxel 0.03 --size 0.1 2>&1 | grep -oE '[0-9]+ region\(s\)' | head -1)
    if [ "$got" = "$want region(s)" ]; then
        ok "overlap $r"
    else
        bad "overlap $r" "'$got', want $want region(s)"
    fi
done

if "$exe" --shape sphere --dim 24 --overlap nosuchrule > /dev/null 2>&1; then
    bad bad-overlap "a bad --overlap was accepted"
else
    ok bad-overlap
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
