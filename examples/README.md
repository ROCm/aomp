Examples 
========

The LLVM compiler supports several accelerated programming models.
Examples in this directory demonstrate how to use the LLVM compiler to support
these different programming models and the tools to support them. 

The examples are divided into categories [openmp](openmp), [hip](hip), 
[fortran](fortran), [cloc](cloc), [raja](raja), [kokkos](kokkos), and [tools](tools).
Each category is a directory and each example
is a subdirectory of that directory. Typically there is a Makefile 
for each example, so the commands
```
make
make run
```
will compile and execute the example.

Each Makefile typically includes the file 
[Makefile.find_gpu_and_install_dir](Makefile.find_gpu_and_install_dir).
This include file will attempt to find the LLVM compiler using the
environment variable LLVM_INSTALL_DIR. If that is not preset, or the
value specifies a nonexistant directory, it searches this list of directories:

```
  $(HOME)/rocm/aomp /opt/rocm/llvm /usr/lib/aomp
```
It sets LLVM_INSTALL_DIR to the first directory it finds. 
It is recommended that users of these examples preset LLVM_INSTALL_DIR
to avoid this search.

The include file then determines if there is an active GPU and
the GPU architecture (eg. gfx940, sm_70, etc).
It first checks for an active amdgpu kernel module
If there is an active amdgpu kernel, the LLVM compiler utility
$(LLVM_INSTALL_DIR)/bin/amdgpu-arch is used to set 
the value LLVM_GPU_ARCH which is needed by the example Makefiles.
If no active amdgpu kernel is found, the include file looks for an nvidia PCI
device and then uses the compiler utility \$(LLVM_INSTALL_DIR)/bin/amdgpu-arch
to set the value LLVM_GPU_ARCH.

If no LLVM installation or GPU could be found, the example Makefile will fail. 

There are often other make targets to show different ways to build the binary.
E.g. to run with some debug output set OFFLOAD_DEBUG variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

Run ```make help``` to see all the possible demos of each example. 
