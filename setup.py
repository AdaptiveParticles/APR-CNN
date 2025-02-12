# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        build_type = "Debug" if self.debug else "Release"

        build_args = ['--config', build_type]

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(build_type),  # not used on MSVC, but no harm
        ]

        # Pass CMake arguments via the environment variable 'EXTRA_CMAKE_ARGS'
        cmake_args.extend([x for x in os.environ.get('EXTRA_CMAKE_ARGS', '').split(' ') if x])

        if self.compiler.compiler_type == "msvc":

            # Must provide -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
            # via environment variable EXTRA_CMAKE_ARGS
            cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            cmake_args += ["-DVCPKG_TARGET_TRIPLET=x64-windows"]
            cmake_args += ["-T"]
            cmake_args += ["ClangCL"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print("******************************************************************************")

        print( ext.sourcedir)
        print("*******************************CMAKE ARGS***********************************************")
        print(cmake_args)
        print( self.build_temp)

        print(["cmake", ext.sourcedir] + cmake_args)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel", "8"] + build_args, cwd=self.build_temp
        )



setup(
    ext_modules=[CMakeExtension('_pyaprwrapper')],
    cmdclass={
        'build_ext': CMakeBuild,
    }
)
