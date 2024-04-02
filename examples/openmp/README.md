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
/usr/lib/aomp/bin/clang -O3 -target x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 reduction.c -o reduction

:/usr/lib/aomp/examples/openmp/reduction$ make run
./reduction
The result is correct = 499999500000!

```

