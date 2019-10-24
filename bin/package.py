# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------

from spack import *


class Aomp(MakefilePackage):
    """  llvm openmp compiler from AMD"""

    homepage = "https://github.com/ROCm-Developer-Tools/aomp"
    url      = "https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp-0.7-5.tar.gz"

    version('0.7-5', sha256='8f3b20e57bf2032d388879429f29b729ce9a46bee5e7ba76976fc77ea48707a7')

    family = 'compiler'

    def edit(self, spec, prefix):
        makefile = FileFilter('Makefile')

    def install(self, spec, prefix):
        make()
        make("install")
