#!/bin/bash
set -x

export AOMP
 $AOMP/bin/flang \
    -DCLOUDSC_GPU_OMP_SCC -DCLOUDSC_GPU_SCC -DCLOUDSC_STMT_FUNC -DHAVE_HDF5 \
    -fpic -fPIE \
    -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
    -c cloudsc_gpu_omp_scc_mod.F90 \
    -o cloudsc_gpu_omp_scc_mod.F90.o 


 $AOMP/bin/flang \
    -DCLOUDSC_GPU_OMP_SCC -DCLOUDSC_GPU_SCC -DCLOUDSC_STMT_FUNC -DHAVE_HDF5 \
    -fpic -fPIE \
    -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
    -c cloudsc_driver_gpu_omp_scc_mod.F90 \
    -o cloudsc_driver_gpu_omp_scc_mod.F90.o 
    
  rm -rf *.mod *.ilm *.cmdx *.cmod *.o *.stb

    
