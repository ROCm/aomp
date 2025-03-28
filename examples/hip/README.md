AOMP - Examples to demonstrate use of HIP
=========================================

The ROCm compiler supports several accelerated programming models.
This includes a HIP front end. This allows HIP code to be compiled
by the ROCm compiler, which will produce a binary containing code
generated for the specified target GPU, as well as host code. The
examples in this directory demonstrate how to use AOMP to compile HIP
code to an executable.

To compile an example simply run 'make', and to run 'make run'. For
more information and other options use 'make help'

Examples:
- [vectorAdd](vectorAdd/README.md) Simple demo to add two vectors
- [writeIndex](writeIndex/README.md) Write the thread index into an array element
- [matrixmul](matrixmul/README.md) simple implementation of matrix multiplication
- [structWithPointersUM](structWithPointersUM/README.md) Demo use of unified memory in HIP
- [usm](usm/README.md) Demo of building and use heterogeneous static device library (SDL)
- [device_lib](device_lib/README.md) Demo building and using heterogeneous static device library (SDL)
for HIP. Two SDLs are built. The first (_mylib) is built manually. The second SDL (_myomplib) is
built with openmp target offload.  This is much easier to build than manually.
Currently hip packages heterogenous objects as bundles whereas
OpenMP uses the newer packaged format.  So this demo uses a utility to convert
the packaged archive to bundled archive. 

### About this file

This is the README.md file for
https:/github.com/ROCM-Developer-Tools/aomp/examples/hip

