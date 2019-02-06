AOMP - Examples to demonstrate use of Clang CUDA in AOMP
=============================================================

The AOMP compiler supports several accelerated programming models.
AOMP includes native Clang CUDA supprt. This allows CUDA code to be
compiled by the AOMP compiler for Nvidia architecture.
The examples in this directory demonstrate how to use AOMP to
compile CUDA kernel code to an executable.

To compile an example simply run 'make', and to run 'make run'.
To specify GPU architecture AOMP_GPU env variable can be used,
e.g. AOMP_GPU=sm_35.
For more information and other options use 'make help'

Examples:
vectorAdd  - adds two vectors
writeIndex - Write the thread index into an array element
matrixmul  - simple implementation of matrix multiplication

### About this file

This is the README.md file for
https:/github.com/ROCM-Developer-Tools/aomp/examples/cuda

