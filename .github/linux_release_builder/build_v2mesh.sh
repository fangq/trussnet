#!/bin/bash
# Builds the portable Linux v2mesh binary inside the manylinux2014 image
# (CentOS 7, glibc 2.17; after mcx's .github/linux_release_builder):
#
#   docker run --rm -v $PWD:/src -w /src quay.io/pypa/manylinux2014_x86_64 \
#       bash .github/linux_release_builder/build_v2mesh.sh
#
# Linked here the binary needs at most GLIBC_2.17 and starts on any distribution
# since (Ubuntu 14.04, CentOS 7); built on the ubuntu-22.04 runner it needed
# GLIBC_2.35. glibc stays dynamic (a static one cannot dlopen the OpenCL ICD
# loader); libstdc++, libgcc and libgomp are linked statically. The CLI tests
# run on the result, and the build fails if it needs a newer glibc.
#
# Run with "docker run", not as a job container: the JavaScript actions
# (checkout, upload-artifact) need node 20, which needs glibc 2.28.

set -e
cd /src

# the headers only: libOpenCL is loaded at run time
yum install -y opencl-headers

# git refuses a checkout owned by another user, which a mounted one is
git config --global --add safe.directory '*' 2>/dev/null || true

BUILD_DIR=build-manylinux
rm -rf $BUILD_DIR
# the static libgomp after the objects; --as-needed then drops the shared one
# -fopenmp adds after it
make check BUILD_DIR=$BUILD_DIR CMAKE_ARGS="-DV2M_STATIC_LINK=ON \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--as-needed \
    '-DCMAKE_CXX_STANDARD_LIBRARIES=-Wl,-Bstatic -lgomp -Wl,-Bdynamic -lpthread -ldl'"

strip $BUILD_DIR/v2mesh
bash .github/linux_release_builder/check_glibc.sh $BUILD_DIR/v2mesh
$BUILD_DIR/v2mesh --shape sphere --dim 24 --gpu 2>&1 | grep 'running on the CPU'

# the runner's user owns the checkout
chown -R "$(stat -c %u:%g /src)" $BUILD_DIR
