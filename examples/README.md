Compiler Examples
=================

The LLVM compiler supports several accelerated programming models
including OpenMP, HIP, opencl, stdpar, hipfort, raja, and kokkos.
Acceleration with OpenMP is available for FORTRAN, c, and C++.
These Compiler examples demonstrate how to use the LLVM compiler to compile,
link, and execute applications using different programming models and
various utilities to support development. Each Compiler example uses
'make' to clearly show the full commands to compile, link, and execute
an application. The execution of make will also show any supporting commands
and environment variable settings needed to support the example.

## Compiler Example Categories

The Compiler examples are divided into categories:

- [OpenMP](openmp/README.md)  C++ and c examples of OpenMP target offload
- [FORTRAN](fortran)  FORTRAN examples of OpenMP target offload
- [HIP](hip/README.md)  Acceleration with HIP
- [Raja](raja/README.md)  Examples that use the RAJA programming model
- [Kokkos](kokkos)  Examples that use the kokkos programming model
- [tools](tools)  Examples on how to use profiling, debug, and address sanitizer tools
- [stress](stress)  Examples that are complex or stress resources. CI is not run on these examples
- [stdpar](stdpar)  C++ stdpar examples

Eventually we will add these categories:

- [opencl](opencl)  Examples of offloading with OpenCL
- [hipfort](hipfort)  FORTRAN examples that use FORTRAN HIP bindings

Each category is a directory and each example is a subdirectory of the category directory.


## Using Compiler Example Makefiles
 
There is a Makefile for each Compiler example. The Makefile(s) can be executed
with update access to the example directory where the Makefile and sources are
found, "in-tree", OR from a different directory, "out-of-tree".
These commands demonstrate how to test the example "in-tree":
```
cd openmp/veccopy
make run

```
An "out-of-tree" test is useful when examples are stored in a read-only
directory such as an installation directory. For example, to run the
veccopy example "out-of-tree" from a new /tmp directory run these commands:
```
mkdir /tmp/demo ; cd /tmp/demo
EXROOT=/opt/rocm/share/examples/Compiler  # The examples base directory.
make -f $EXROOT/openmp/veccopy/Makefile run
```
The above will compile and execute the openmp/veccopy example.
The Makefile will print the compile and execute commands to the console.

If you plan to copy the sources for these examples, be sure to copy
the entire Compiler examples directory.  For example:
```
EXROOT=/opt/rocm/share/examples/Compiler  # The examples base directory.
cp -rp $EXROOT /tmp
cd /tmp/Compiler/openmp/veccopy
make run 
```
## Setting LLVM_INSTALL_DIR and LLVM_GPU_ARCH

Each Makefile includes the common Makefile helper file
[inc/find_gpu_and_install_dir.mk](inc/find_gpu_and_install_dir.mk).
This "include-file" finds the LLVM compiler and sets environment
variable `LLVM_INSTALL_DIR` if not already preset by the user.
If not preset, or `LLVM_INSTALL_DIR` specifies a nonexistant directory,
the include-file searches this list of directories to set 'LLVM_INSTALL_DIR':

```
/opt/rocm/lib/llvm /usr/lib/aomp ./../../../llvm  ~/rocm/aomp
```
If no LLVM installation is found, all Compile examples will fail.
Therefore, it is recommended that users of these examples preset `LLVM_INSTALL_DIR`
to demo or test a compiler other than the last installed ROCm compiler
found at /opt/rocm/lib/llvm. 

The include-file then searches for an active GPU to set `LLVM_GPU_ARCH`
(e.g. gfx90a, gfx940, sm_70, etc).
Each example uses the value of `LLVM_GPU_ARCH` to instruct the compiler
which GPU to for compilation. If `LLVM_GPU_ARCH` is not preset,
the include-file first checks for an active amdgpu kernel module.
If there is an active amdgpu kernel module, the include-file calls the LLVM compiler utility
`$(LLVM_INSTALL_DIR)/bin/amdgpu-arch` to set the value of `LLVM_GPU_ARCH`.
If no active amdgpu kernel module is found, the include file looks for an nvidia PCI
device and then uses the compiler utility `$(LLVM_INSTALL_DIR)/bin/nvptx-arch`
to set the value `LLVM_GPU_ARCH`.
If there are no active GPUs or the examples are being run for compile-only,
users may preset `LLVM_GPU_ARCH` to avoid this search.

If `LLVM_GPU_ARCH` is not preset and no GPUs are found, all Compile examples will fail.

## Makefile Targets

The default for each Makefile is to compile and link the example binary.
The target "run" will compile, link, and execute the example.
The target "clean" will remove all generated files by the Makefile.   

There are often other make targets. To show different ways to build the binary.
E.g. to run with some debug output set `OFFLOAD_DEBUG` variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

Run ```make help``` to see various demos of each example.

## Contributions

If you want to contribute a Compiler example, please read these [rules](inc/contribute_rules.md).
