Examples 
========

The LLVM compiler supports several accelerated programming models
including openmp, hip, opencl, stdpar, raja, and kokkos. 
OpenMP is available for FORTRAN, c, and c++. 
Examples in this directory demonstrate how to use the LLVM compiler to support
these different programming models and the tools to support them. 

The examples are divided into categories [openmp](openmp), [hip](hip), 
[fortran](fortran), [raja](raja), [kokkos](kokkos), [tools](tools) and [stress](stress).
Eventually we will add categories  [stdpar](stdpar) and [opencl](opencl).
Each category is a directory and each example is a subdirectory of the category directory. 
Typically there is a Makefile for each example, so the commands
```
EXAMPLE_ROOT=$PWD  # The examples base directory.
mkdir /tmp/demo ; cd /tmp/demo
make -f $EXAMPLE_ROOT/openmp/veccopy/Makefile  run 
```
will compile and execute the openmp/veccopy example. 
The Makefile will print the commands executed to your console.

Each Makefile typically includes the common Makefile helper file
[Makefile.find_gpu_and_install_dir](Makefile.find_gpu_and_install_dir).
This include file finds the LLVM compiler and sets the environment 
variable LLVM_INSTALL_DIR if not already preset by the user. 
If not preset, or the value of LLVM_INSTALL_DIR specifies a nonexistant directory, 
the Makefile searches this list of directories:

```
/opt/rocm/lib/llvm  /usr/lib/aomp  ./../../../llvm  ~/rocm/aomp
```
The Makefile sets LLVM_INSTALL_DIR to the first directory it finds in the above list.
It is recommended that users of these examples preset LLVM_INSTALL_DIR
if they want to demo the compiler with something other than the last
installed ROCm compiler. 

The include file then determines if there is an active GPU and
the GPU architecture (eg. gfx940, sm_70, etc).
It first checks for an active amdgpu kernel module
If there is an active amdgpu kernel, the LLVM compiler utility
\$(LLVM_INSTALL_DIR)/bin/amdgpu-arch is used to set 
the value LLVM_GPU_ARCH which instructs the compiler which GPU to compile for. 
If no active amdgpu kernel is found, the include file looks for an nvidia PCI
device and then uses the compiler utility \$(LLVM_INSTALL_DIR)/bin/amdgpu-arch
to set the value LLVM_GPU_ARCH.

If no LLVM installation or GPU could be found, the example Makefile will typically fail. 

There are often other make targets to show different ways to build the binary.
E.g. to run with some debug output set OFFLOAD_DEBUG variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

Run ```make help``` to see all the possible demos of each example. 
