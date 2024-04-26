Examples
========

The LLVM compiler supports several accelerated programming models
including OpenMP, HIP, opencl, stdpar, hipfort, raja, and kokkos.
Acceleration with OpenMP is available for FORTRAN, c, and C++.
These examples demonstrate how to use the LLVM compiler to compile and execute
applications using these different programming models and various utilities to
support development.

The examples are divided into categories. These categories are:

- [OpenMP](openmp/README.md)  C++ and c examples of OpenMP target offload
- [FORTRAN](fortran)  FORTRAN examples of OpenMP target offload
- [HIP](hip/README.md)  Acceleration with HIP
- [Raja](raja/README.md)  Examples that use the RAJA programming model
- [Kokkos](kokkos)  Examples that use the kokkos programming model
- [tools](tools)  Examples on how to use profiling, debug, and address sanitizer tools
- [stress](stress)  Examples that are complex or stress resources

Eventually we will add these categories:

- [stdpar](stdpar)  C++ stdpar examples
- [opencl](opencl)  Examples of offloading with OpenCL
- [hipfort](hipfort)  FORTRAN examples that use FORTRAN HIP bindings

Each category is a directory and each example is a subdirectory of the category directory.
Typically there is a Makefile for each example. That Makefile can be executed with either update
access to the example directory or in a different directory since examples typically exist in a
read-only installation directory. These commands show how to run an example from
a writable tmp directory.
```
mkdir /tmp/demo ; cd /tmp/demo
EXROOT=/opt/rocm/share/openmp-extras/examples  # The examples base directory.
make -f $EXROOT/openmp/veccopy/Makefile run
```
The above will compile and execute the openmp/veccopy example.
The Makefile will print the compile and execute commands to your console.

Each Makefile typically includes the common Makefile helper file
[Makefile.find_gpu_and_install_dir](Makefile.find_gpu_and_install_dir).
This include file finds the LLVM compiler and sets the environment
variable `LLVM_INSTALL_DIR` if not already preset by the user.
If not preset, or the value of `LLVM_INSTALL_DIR` specifies a nonexistant directory,
the Makefile searches this list of directories:

```
/opt/rocm/lib/llvm /usr/lib/aomp ./../../../llvm  ~/rocm/aomp
```

The Makefile sets `LLVM_INSTALL_DIR` to the first directory it finds in the above list.
It is recommended that users of these examples preset `LLVM_INSTALL_DIR`
if they want to demo the compiler with something other than the last
installed ROCm compiler.

The include file then determines if there is an active GPU and
the GPU architecture (eg. gfx940, sm_70, etc).
It first checks for an active amdgpu kernel module
If there is an active amdgpu kernel, the LLVM compiler utility
`$(LLVM_INSTALL_DIR)/bin/amdgpu-arch` is used to set
the value `LLVM_GPU_ARCH` which instructs the compiler which GPU to compile for.
If no active amdgpu kernel module is found, the include file looks for an nvidia PCI
device and then uses the compiler utility `$(LLVM_INSTALL_DIR)/bin/amdgpu-arch`
to set the value `LLVM_GPU_ARCH`.

If no LLVM installation or GPU could be found, the example Makefile will fail.
So if you want to demo the compiler with something other than the last
installed ROCm compiler (which is typically found at /opt/rocm/lib/llvm),
then it is recommended that users preset LLVM_INSTALL_DIR.

There are often other make targets to show different ways to build the binary.
E.g. to run with some debug output set `OFFLOAD_DEBUG` variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

Run ```make help``` to see all the possible demos of each example.
