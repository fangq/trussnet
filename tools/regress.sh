#!/bin/bash
# one summary line per synthetic shape: conformity, worst per-label volume error,
# quality, wall time. usage: tools/regress.sh [dim] [extra trussnet options]
dim=${1:-96}; shift
exe=$(dirname "$0")/../build/trussnet
for s in sphere twoballs corrsphere gyroid torus ushape tjunction helix3 boxhemi sandwich shells hollow slab wedge holeysheet; do
    t0=$(date +%s.%N)
    out=$($exe --shape $s --dim $dim "$@" 2>&1)
    t1=$(date +%s.%N)
    conf=$(echo "$out" | grep "tess\]" | sed 's/.*conformity: //; s/ repair rounds.*//; s/ bad faces/bf/; s/ edges through label 0/ out/; s/ spanning/ span/')
    vol=$(echo "$out" | grep "per-label" | sed 's/.*(max |\([0-9.]*\)%|).*/\1/')
    dev=$(echo "$out" | grep "voxel interface" | sed 's/.*faces \([0-9.\/]*\), spanning \([0-9.\/]*\)/\1 \2/' | awk '{split($1,a,"/"); split($2,b,"/"); m=a[4]>b[4]?a[4]:b[4]; print m}')
    q=$(echo "$out" | grep "qual\]" | sed 's/.*p5 \([0-9.]*\) median \([0-9.]*\).*/\1 \2/')
    printf "%-11s %-26s vol %5s%%  dev %5s  JL p5/med %s  %5.1f s\n" $s "$conf" "$vol" "${dev:-0}" "$q" $(echo "$t1 - $t0" | bc)
done
