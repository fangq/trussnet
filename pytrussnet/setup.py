# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- pip build of the Python module: runs the repository-root CMake with
# -DTN_BUILD_PYTHON=ON (compiling the core, the exact Delaunay and, if an OpenCL
# ICD loader is present, the GPU path) and packs pytrussnet/trussnet/_trussnet*.so.
# TN_USE_OPENCL=OFF builds a CPU-only module.
import os
import shutil
import subprocess
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
            "-DTN_BUILD_PYTHON=ON",
            f"-DTN_USE_OPENCL={os.environ.get('TN_USE_OPENCL', 'ON')}",
        ]
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
                "_trussnet",
                "--config",
                cfg,
                "--parallel",
            ],
            check=True,
        )
        built = glob(str(HERE / "trussnet" / "_trussnet*"))
        if not built:
            raise RuntimeError("trussnet: the CMake build produced no _trussnet module")
        dst = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(built[0], dst / Path(built[0]).name)


setup(
    packages=["trussnet"],
    ext_modules=[CMakeExtension("trussnet._trussnet")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
