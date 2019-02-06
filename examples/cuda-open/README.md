AOMP - Examples to demonstrate use of Clang HIP/CUDA in AOMP
=============================================================

The AOMP compiler supports several accelerated programming models.
AOMP includes a HIP/CUDA kernel front end. This allows CUDA code to be
compiled by the AOMP compiler, which will produce a binary containing
code generated for the specified target GPU, as well as host code
using the HIP host API. The examples in this directory demonstrate how
to use AOMP to compile CUDA kernel code to an executable.

To compile an example simply run 'make', and to run 'make run'. For
more information and other options use 'make help'

Examples:
vectorAdd  - adds two vectors
writeIndex - Write the thread index into an array element
matrixmul  - simple implementation of matrix multiplication

### About this file

This is the README.md file for
https:/github.com/ROCM-Developer-Tools/aomp/examples/cuda-open

