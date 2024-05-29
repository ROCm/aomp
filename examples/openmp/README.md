Examples to demonstrate use of OpenMP in c and C++
==================================================

The LLVM compiler supports several accelerated programming models for GPUs.
OpenMP is one of these programming models used to accelerate c, C++ and FORTRAN.
Examples in these subdirectories demonstrate how to compile and execute c and C++ 
sources to use OpenMP target offload.

These are the c and C++ examples in the [openmp](.) examples category:
- [hello](hello/README.md) A printf demo showing various levels of parallelism
- [veccopy](veccopy) A simple vector copy in c
- [vmulsum](vmulsum/README) A simple vector multiply and sum in c
- [reduction](reduction/README) A simple parallel reduction in c
- [print_device](print_device/README.md) Prints the number of GPUs and the value of ROCR_VISIBLE_DEVICES to demonstrate gpurun.
- [driver_tests](driver_tests) Demo of various command line options
- [declare_variant_if](declare_variant_if)
- [demo_offload_types](demo_offload_types) Demo showing use of OMP_TARGET_OFFLOAD in various build types. 
- [vmul_template](vmul_template/README) A C++ example
- [atomic_vs_reduction](atomic_vs_reduction/README.md) Demo showing it's better to use a reduction than atomic update in a parallel loop. 
