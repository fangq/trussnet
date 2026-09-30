# SPDX-License-Identifier: GPL-3.0-or-later
# v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
#
# Convenience wrapper around the CMake build.
#   make            configure + build under build/ (OpenCL + CDT on)
#   make cpu        OpenCL off
#   make check      build with the CLI tests and run them (ctest)
#   make bindings   also the Python module and the MATLAB / Octave MEX (build-bind/)
#   make test       their unit tests (ctest)
#   make pretty     auto-format: astyle (C++), black (Python), mh_style (MATLAB)
#   make clean
# cmake by its full path: with "." (or an empty entry) in PATH, make would run
# the cmake/ directory here and stop ("cmake: Permission denied")
CMAKE      ?= $(shell command -v cmake 2>/dev/null || echo cmake)
BUILD_DIR  ?= build
BUILD_TYPE ?= Release

all:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	$(CMAKE) --build $(BUILD_DIR) --parallel

cpu:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DV2M_USE_OPENCL=OFF
	$(CMAKE) --build $(BUILD_DIR) --parallel

BIND_DIR ?= build-bind
MATLAB_ROOT ?= $(shell dirname $$(dirname $$(readlink -f $$(which matlab 2>/dev/null) 2>/dev/null)) 2>/dev/null)

check:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DV2M_BUILD_TESTS=ON $(CMAKE_ARGS)
	$(CMAKE) --build $(BUILD_DIR) --parallel
	cd $(BUILD_DIR) && ctest --output-on-failure

bindings:
	$(CMAKE) -S . -B $(BIND_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DV2M_BUILD_TESTS=ON -DV2M_BUILD_PYTHON=ON \
	      -DV2M_BUILD_MATLAB_MEX=ON -DV2M_BUILD_OCTAVE_MEX=ON $(if $(MATLAB_ROOT),-DMatlab_ROOT_DIR=$(MATLAB_ROOT))
	$(CMAKE) --build $(BIND_DIR) --parallel

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
PRETTY_CPP := src/v2m_pipeline.cpp src/v2m_pipeline.h src/v2m_mex.cpp src/pyv2mesh.cpp \
              src/v2m_gdel.cpp src/v2m_gdel.h src/v2m_tpm.cpp src/v2m_tpm.h \
              src/v2m_2d.cpp src/v2m_2d.h src/v2m_isize.h src/v2m_plc.cpp src/v2m_plc.h \
              src/v2m_step.cpp src/v2m_step.h

pretty:
	astyle $(ASTYLE_FLAGS) $(PRETTY_CPP)
	python3 -m black -l 100 pyv2mesh tools/mkgray_jacobian.py tools/v2mslice.py tools/v2mcut.py
	mh_style --fix matlab

clean:
	rm -rf $(BUILD_DIR) $(BIND_DIR)

.PHONY: all cpu check bindings test pretty clean
