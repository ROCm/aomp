Example: cloc/vector_copy 
=========================

This example shows how to use the cloc.sh script to offline compile an OpenCL kernel 
in vector_copy.cl into and HSA Code Object (hsaco). The source code in vector_copy.cpp
uses the HSA API to launch the kernel found in the hsaco. 

To compile and run this example, execute these commands:

```
  make
  ./vector_copy
```

The Makefile will execute these commands to build the vector_copy binary:


```
g++ -c -std=c++11 -I/opt/rocm/aomp/include -o obj/vector_copy.o vector_copy.cpp
g++ obj/vector_copy.o -L/opt/rocm/aomp/lib -lhsa-runtime64 -Wl,-rpath=/opt/rocm/aomp/lib -o vector_copy
/opt/rocm/aomp/bin/cloc.sh -mcpu gfx803 vector_copy.cl

```
The vectory_copy program will load the GPU binary in the file vector_copy.hsaco.
To create this file with cloc.sh, the Makefile runs the cloc.sh utility as the last command.
```


