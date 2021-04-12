# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------

from spack import *


class Aomp(MakefilePackage):
    """  llvm openmp compiler from AMD"""

    homepage = "https://github.com/ROCm-Developer-Tools/aomp"
    url      = "https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_13.0-2/aomp-13.0-2.tar.gz"

    # Fixme: this will be adjusted when spack create is called. When copying over to your own package.py omit this line.
    version('13.0-2', sha256='7ea4e218b171585441278f3562d97779006b12bc3b5dc201901f2d757226da84')

    family = 'compiler'

    def edit(self, spec, prefix):
        makefile = FileFilter('Makefile')
        filter_file('add_subdirectory(test)', '#add_subdirectory(test)','amd-llvm-project/compiler-rt/CMakeLists.txt', string=True)
        filter_file('add_subdirectory(test)', '#add_subdirectory(test)','amd-llvm-project/llvm/CMakeLists.txt', string=True)
        filter_file('add_subdirectory(test)', '#add_subdirectory(test)','flang/CMakeLists.txt', string=True)

    def install(self, spec, prefix):
        make()
        make("install")
