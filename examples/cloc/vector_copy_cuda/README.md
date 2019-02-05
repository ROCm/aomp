Example: cloc/vector_copy_cuda
==============================

This example shows how to use the cloc.sh script to offline compile cuda kernels
in vector_copy.cu into and HSA Code Object (hsaco). The source code in vector_copy.cpp
uses the HSA API to load and launch the kernel from the hsaco file. 

To compile and run this example, execute these commands:

```
  make
  ./vector_copy
```

The Makefile will execute these commands to build the vector_copy binary:

```
  g++ -I/opt/rocm/include -c -std=c++11 -o obj/vector_copy.o vector_copy.cpp
  g++  obj/vector_copy.o -L/opt/rocm/lib -lhsa-runtime64 -o vector_copy
```

The vectory_copy program will look for the file vector_copy.hsaco.
To create hsaco with cloc.sh the Makefile will run the cloc.sh utility
as follows:

```
  /opt/rocm/aomp/bin/cloc.sh vector_copy.cu
```
