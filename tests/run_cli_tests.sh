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

if run "gpu (or its CPU fallback)" --shape sphere --dim 48 --gpu; then
    conforming && ok "gpu (or its CPU fallback)" || bad "gpu (or its CPU fallback)" "not conforming"
fi

if run "voxel trapping" --shape sphere --dim 48 --trap voxel; then
    ok "voxel trapping"
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
