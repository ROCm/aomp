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
  g++ -I/opt/rocm/include -c -std=c++11 -o obj/vector_copy.o vector_copy.cpp
  g++  obj/vector_copy.o -L/opt/rocm/lib -lhsa-runtime64 -o vector_copy
```

The vectory_copy program will look for the file vector_copy.hsaco.
To create this file with cloc.sh, the Makefile runs the cloc.sh utility
as follows:

```
  /opt/rocm/aomp/bin/cloc.sh vector_copy.cl
```


