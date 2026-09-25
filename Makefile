# Convenience wrapper around the CMake build.
#   make            configure + build under build/ (OpenCL + CDT on)
#   make cpu        OpenCL off
#   make clean
BUILD_DIR  ?= build
BUILD_TYPE ?= Release

all:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

cpu:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DTN_USE_OPENCL=OFF
	cmake --build $(BUILD_DIR) --parallel

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all cpu clean
