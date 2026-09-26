# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
#
# Convenience wrapper around the CMake build.
#   make            configure + build under build/ (OpenCL + CDT on)
#   make cpu        OpenCL off
#   make check      build with the CLI tests and run them (ctest)
#   make bindings   also the Python module and the MATLAB / Octave MEX (build-bind/)
#   make test       their unit tests (ctest)
#   make pretty     auto-format: astyle (C++), black (Python), mh_style (MATLAB)
#   make clean
BUILD_DIR  ?= build
BUILD_TYPE ?= Release

all:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

cpu:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTN_USE_OPENCL=OFF
	cmake --build $(BUILD_DIR) --parallel

BIND_DIR ?= build-bind
MATLAB_ROOT ?= $(shell dirname $$(dirname $$(readlink -f $$(which matlab 2>/dev/null) 2>/dev/null)) 2>/dev/null)

check:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTN_BUILD_TESTS=ON $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) --parallel
	cd $(BUILD_DIR) && ctest --output-on-failure

bindings:
	cmake -S . -B $(BIND_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTN_BUILD_TESTS=ON -DTN_BUILD_PYTHON=ON \
	      -DTN_BUILD_MATLAB_MEX=ON -DTN_BUILD_OCTAVE_MEX=ON $(if $(MATLAB_ROOT),-DMatlab_ROOT_DIR=$(MATLAB_ROOT))
	cmake --build $(BIND_DIR) --parallel

test: bindings
	cd $(BIND_DIR) && ctest --output-on-failure

# astyle settings of MCX / gpu_brain2mesh. Applied to the files kept astyle-clean
# (astyle 3.1 mis-parses a few idioms of the older sources, e.g. a product in a
# brace initializer, so those are not reformatted wholesale); the .cl kernels
# and third_party/ are never reformatted.
ASTYLE_FLAGS := --style=attach --indent=spaces=4 --indent-modifiers \
                --indent-switches --indent-preproc-block --indent-preproc-define \
                --indent-col1-comments --pad-oper --pad-header --align-pointer=type \
                --align-reference=type --add-brackets --convert-tabs --close-templates \
                --lineend=linux --preserve-date --suffix=none --formatted --break-blocks
PRETTY_CPP := src/tn_pipeline.cpp src/tn_pipeline.h src/tn_mex.cpp src/pytrussnet.cpp \
              src/tn_gdel.cpp src/tn_gdel.h src/tn_tpm.cpp src/tn_tpm.h \
              src/tn_2d.cpp src/tn_2d.h src/tn_isize.h

pretty:
	astyle $(ASTYLE_FLAGS) $(PRETTY_CPP)
	python3 -m black -l 100 pytrussnet tools/mkgray_jacobian.py tools/tnslice.py tools/tncut.py
	mh_style --fix matlab

clean:
	rm -rf $(BUILD_DIR) $(BIND_DIR)

.PHONY: all cpu check bindings test pretty clean
