AOMP - Examples to demonstrate use of RAJA with omp_target
==========================================================

The AOMP compiler supports several accelerated programming
models. RAJA is one of these models. Examples in this directory
demonstrate how to use AOMP to compile OpenMP 4.5 sources using the
RAJA library, and execute the binaries on GPU.

Cd to a specific example folder and run the following commands to
build and execute:

```
make
make run
```

If RAJA is not built, the test Makefile will automatically configure
and build RAJA first. The build directory prefix can be specified by
setting the RAJA_BUILD_PREFIX (default is $HOME) environment
variable. The build script used to configure and build RAJA is located
in $AOMP/bin/raja_build.sh

Run ```make help``` to see on how to clean the build, including RAJA, and other options. 

E.g. to run with some debug output set OFFLOAD_DEBUG variable:

```
env OFFLOAD_DEBUG=1 make
env OFFLOAD_DEBUG=1 make run
```

This demonstration shows the commands and output for the vecadd example:

```
:/usr/lib/aomp/examples/raja/vecadd$ make
/usr/lib/aomp/bin/clang++ -w -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -I/home/alit/git/aomp/raja/include  -I/home/alit/raja_build.gfx906/include   vecadd.cpp -o vecadd

:/usr/lib/aomp/examples/raja/vecadd$ make run
./vecadd
Success

```

