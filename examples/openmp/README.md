AOMP - Examples to demonstrate use of OpenMP
==============================================

The AOMP compiler supports several accelerated programming models. OpenMP is one of these models. 

Examples in this directory demonstrate how to use aomp to compile OpenMP 4.5 sources and execute the binaries on GPU.

Cd to a specific example folder and run the following commands to build and execute:

```
make
make run
```
There are many other make targets to show different ways to build the binary. Run ```make help``` to see all the possible demos as Makefile targets.

E.g. to run with some debug output set OFFLOAD_DEBUG variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

This demonstration shows the commands and output for the reduction example:

```
:/usr/lib/aomp/examples/openmp/reduction$ make
/usr/lib/aomp/bin/clang -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 reduction.c -o reduction

:/usr/lib/aomp/examples/openmp/reduction$ make run
./reduction
The result is correct = 499999500000!

```

## Bundling

This demonstration shows how intermediate .o object files are actually bundles of multiple files. A .o bundle contains a host object file and an object file for each offload architecture. The unbundle.sh bash script provided with AOMP examines the bundled .o file to create the proper call to the clang-offload-bundler tool.

The makefiles provided in the openmp examples demonstrate two build schemes. The default make creates the executable with a single command as shown above.  The make target "make obin" compiles each source into .o files and then links these into the executable "obin".  In this demonstration, we create the intermediate .o files and the obin executable. Then we use the unbundle.sh script to show the contents of just one of the .o files. Then we rebuild that .o file for Nvidia sm_30 GPUs to show how the .o bundle is different. 

```
:/usr/lib/aomp/examples/openmp/vmulsum$ ls
main.c  Makefile  README  vmul.c  vsum.c

# Compile each source into .o and then link .o files to create a binary
:/usr/lib/aomp/examples/openmp/vmulsum$ make obin
/usr/lib/aomp/bin/clang -c -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900     main.c -o main.o
/usr/lib/aomp/bin/clang -c -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900     vsum.c -o vsum.o
/usr/lib/aomp/bin/clang -c -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900     vmul.c -o vmul.o
/usr/lib/aomp/bin/clang -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900      main.o vsum.o vmul.o -o obin

# Notice how the above commands create three .o files and the executable "obin"
:/usr/lib/aomp/examples/openmp/vmulsum$ ls
main.c  main.o  Makefile  obin  README  vmul.c  vmul.o  vsum.c  vsum.o

# Now lets unbundle just one of the three .o files
:/usr/lib/aomp/examples/openmp/vmulsum$ $AOMP/bin/unbundle.sh vmul.o
/usr/lib/aomp/bin/clang-offload-bundler -unbundle -type=bc -inputs=vmul.o -targets=openmp-amdgcn-amd-amdhsa,host-x86_64-pc-linux-gnu -outputs=vmul.o.openmp-amdgcn-amd-amdhsa,vmul.o.host-x86_64-pc-linux-gnu

# Two more files appear from the bundle
:/usr/lib/aomp/examples/openmp/vmulsum$ ls
main.c    obin    vmul.o                           vsum.c
main.o    README  vmul.o.host-x86_64-pc-linux-gnu  vsum.o
Makefile  vmul.c  vmul.o.openmp-amdgcn-amd-amdhsa

# For amdgcn, the .o bundle contains a host ELF object and an LLVM bc 
:/usr/lib/aomp/examples/openmp/vmulsum$ file vmul.o*
vmul.o:                          data
vmul.o.host-x86_64-pc-linux-gnu: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
vmul.o.openmp-amdgcn-amd-amdhsa: LLVM IR bitcode

# Lets switch to building for cuda sm_30 GPUs
:/usr/lib/aomp/examples/openmp/vmulsum$ rm vmul.o*
:/usr/lib/aomp/examples/openmp/vmulsum$ export AOMP_GPU=sm_30
:/usr/lib/aomp/examples/openmp/vmulsum$ make vmul.o
/usr/lib/aomp/bin/clang -c -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_30  vmul.c -o vmul.o

# Notice that the .o bundle now contains host ELF object and a cuda cubin
:/usr/lib/aomp/examples/openmp/vmulsum$ $AOMP/bin/unbundle.sh vmul.o
/usr/lib/aomp/bin/clang-offload-bundler -unbundle -type=o -inputs=vmul.o -targets=openmp-nvptx64-nvidia-cuda,host-x86_64-pc-linux-gnu -outputs=vmul.o.openmp-nvptx64-nvidia-cuda,vmul.o.host-x86_64-pc-linux-gnu

:/usr/lib/aomp/examples/openmp/vmulsum$ file vmul.o*
vmul.o:                            ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
vmul.o.host-x86_64-pc-linux-gnu:   ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
vmul.o.openmp-nvptx64-nvidia-cuda: ELF 64-bit LSB relocatable, NVIDIA CUDA architecture,, not stripped

```
