ROCm Compiler Examples for FORTRAN OpenMP Offloading
====================================================

This directory contains examples using flang to compile and execute examples that use FORTRAN OpenMP target offload.
FORTRAN OpenMP target offload uses OpenMP target pragmas in the source to designate code and data 
that is intended to be accelerated on the GPU.   These are the examples in this category.

- [helloworld](helloworld/README.md) - Demo print from inside and outside of target region.
- [bigloop](bigloop/README.md)
- [simple-offload](simple-offlaod/README.md)
- [gdb-simple](gdb-simple/README.md) - Demo of using the ROCm debugger rocgdb
- [is-initial-device-api](is-initial-device-api/README.md)  Demo check for GPU/CPU inside a target region.
- [fortran-hip-interop](fortran-hip-interop/README.md)  Show hows HIP and FORTRAN target offload can be used together.


