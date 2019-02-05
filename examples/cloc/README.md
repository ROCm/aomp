AOMP - Examples to demonstrate use of cloc.sh
==============================================

The AOMP compiler supports several accelerated programming models.  Launching precompiled GPU kernels written in OpenCL or Cudaclang is one of these models.  The cloc.sh utility does the offline compilation of cl or cu source files to create native GPU binaries for the GPU kernels defined in the source files. Examples in this directory demonstrate how to use the cloc.sh utility to compile GPU kernels to a GPU binary file.

The host source code in these examples show different ways to load and execute the binary. The different APIs expect an HSA code object.  These files typically end in the .hsaco file type. The vector_copy and vector_copy_cuda examples use the low level HSA API.  The vector_copy_hip uses the higer level HIP host API to load and execute kernels. The vector_copy_hip_omp example shows multiple tasks launching GPU kernels using OpenMP and the HIP host API.

The name "cloc" originally meant "cl offline compiler".  Kernels written for OpenCL are typically named with the .cl file extension.   However, thanks to the cudaclang compiler and the amdgcn LLVM backend compiler, code object can be created for cudaclang source files. These typically end in the .cu file extension. The cloc.sh utility will operate on either .cl or .cu files. 
 
### About this file

This is the README.md file for https:/github.com/ROCM-Developer-Tools/aomp/examples/cloc
