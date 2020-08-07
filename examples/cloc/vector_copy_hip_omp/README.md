Example: cloc/vector_copy_hip
==============================

This example shows a highly experimental hybrid mode. 

Hybrid mode combines explicit kernel offloading in the same application as OpenMP target offloading. 
cloc.sh is used to compile a cuda kernel in vectory_copy.cu into the binary file vector_copy.hsaco.
This is an HSA Code Object(hsaco).
Then the AOMP compiler is used to compile vector_copy.cpp. The file vector_copy.cpp contains both an explicit kernel launch 
with the HIP API AND an OpenMP target offload region. 

To compile and run this example, execute these commands:

```
  make
  ./vector_copy
```

The Makefile will execute these commands to build the vector_copy binary:

```
/usr/lib/aomp/bin/clang++ -I/usr/lib/include -c -std=c++11 -D__HIP_PLATFORM_HCC__ -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803     -o obj/vector_copy.o vector_copy.cpp
/usr/lib/aomp/bin/clang++  obj/vector_copy.o -std=c++11 -D__HIP_PLATFORM_HCC__ -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803     -L/usr/lib/lib -lamdhip64 -o vector_copy
```
The vectory_copy program will look for the file vector_copy.hsaco.
To create the hsaco with cloc.sh the Makefile will run the cloc.sh utility as follows:

```
/usr/lib/aomp/bin/cloc.sh vector_copy.cu
```
