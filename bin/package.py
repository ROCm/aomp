# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------

from spack import *


class Aomp(MakefilePackage):
    """  llvm openmp compiler from AMD"""

    homepage = "https://github.com/ROCm-Developer-Tools/aomp"
    url      = "https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-3/aomp-0.7-3.tar.gz"

    version('0.7-3', sha256='9a4971df847a0b0d9913ced5ba19e6574e8823d5b74aaac725945f7360402be1')
    family = 'compiler'

    def edit(self, spec, prefix):
        makefile = FileFilter('Makefile')

    def install(self, spec, prefix):
        make()
        make("install")
