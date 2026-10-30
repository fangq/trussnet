#!/bin/bash
# Builds the portable Linux v2m GUI inside the image of this folder's Dockerfile:
#
#   docker build -t v2m-linux-release-builder .github/linux_release_builder
#   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v $PWD:/src \
#       v2m-linux-release-builder bash .github/linux_release_builder/build_v2m.sh
#
# The build fails if the binary needs a glibc newer than 2.17.

set -e
cd /src/v2m

# the check at the end must only ever see what this run built
make clean
make LAZDIR=/opt/lazarus
strip bin/v2m

cd /src
bash .github/linux_release_builder/check_glibc.sh v2m/bin/v2m
