# SPDX-License-Identifier: GPL-3.0-or-later
# v2mesh -- pip build of the Python module: runs the repository-root CMake with
# -DV2M_BUILD_PYTHON=ON (compiling the core, the exact Delaunay and, if the OpenCL
# headers are found, the GPU path, which loads the OpenCL library at run time) and
# packs pyv2mesh/v2mesh/_v2mesh*.so. V2M_USE_OPENCL=OFF builds a CPU-only module.
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from glob import glob
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DV2M_BUILD_PYTHON=ON",
            f"-DV2M_USE_OPENCL={os.environ.get('V2M_USE_OPENCL', 'ON')}",
            f"-DPython_EXECUTABLE={sys.executable}",  # the interpreter building the wheel
        ]
        # extra CMake arguments (CI: the OpenMP root on macOS, the MinGW toolchain, ...)
        args += shlex.split(os.environ.get("V2M_CMAKE_ARGS", ""))  # quote values with spaces
        try:
            import pybind11

            args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build_temp)] + args, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_temp),
                "--target",
                "_v2mesh",
                "--config",
                cfg,
                "--parallel",
            ],
            check=True,
        )
        # the module of THIS interpreter (the tree may hold other versions' builds)
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        built = glob(str(HERE / "v2mesh" / ("_v2mesh" + suffix)))
        if not built:
            raise RuntimeError("v2mesh: the CMake build produced no _v2mesh%s module" % suffix)
        dst = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(built[0], dst / Path(built[0]).name)


# the license texts travel with the wheels (pybind11's BSD notice, the vendored
# LGPL / Apache / MIT components; CREDITS.md lists them)
_lic = HERE / "v2mesh" / "_licenses"
if (ROOT / "LICENSE").exists():
    _lic.mkdir(exist_ok=True)
    for _f in [ROOT / "LICENSE", ROOT / "CREDITS.md"] + sorted((ROOT / "LICENSES").glob("*")):
        if _f.is_file():
            shutil.copy(_f, _lic / _f.name)

setup(
    packages=["v2mesh"],
    package_data={
        "v2mesh": ["*.dll", "_licenses/*"]
    },  # (Windows wheels: the bundled MinGW runtime DLLs)
    ext_modules=[CMakeExtension("v2mesh._v2mesh")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
