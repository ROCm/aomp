device_lib - Demonstrate use of Static Device Libraries 
=======================================================

This example shows two different methods to build a heterogeneous libraries.
A heterogeneous library supports the execution of functions on both the host and device. 
In fact it can be multiple device types. Typically the source for both host and device are the same.
However, sources can be annotated with #ifdef to change behavior for host and device compilations. 

The first method uses seperate archive files for both host and device. 
We call this a multi-file heterogeneous library.  
In this method, the naming convention is important for the compiler to find the correct
library file for device linkikng.
In this demo the library files names for the first method are lib_mylib.a and lib_mylib-amdgcn.a. 

The 2nd method uses OpenMP to compile the source files to create heterogenous object files
called bundles. These files are then inserted into the library archive file  using traditional 
archive tools with the traditional host filenaming convention. The result of this method is a 
single archive file. Some people call this as a "fat" archive.  
For this demo we will call it a single file heterogeneous library. 
In this demo this library is called _myomplib. Applications use this library with -l_myomplib. 
HIP applications include the header file _myomplib.h to use this library. 
The single file name created for _myomplib is lib_myomplib.a which is traditional host convention.  

In this demo we show the library source code from both c and FORTRAN functions.

A heterogeneous device library can be used by both OpenMP and HIP. In this demo, we are only
showing how the libraries are used by a HIP application. 

HIP applications include the header file _mylib.h to use functions in this library. 
These functions can be used by both host code and device code. 

Compilation of HIP applications must use the the option -l_mylib. Compilation also requires
the -L option to tell the compiler where to find the device libraries. 
In this HIP example, we use both the _mylib and _myomplib library functions that happen to be found in the current directory for the HIP compilation.   There for the hip compilation is
```
hipcc -L$PWD -l_mylib -l_myomplib writeIndex.hip -o writeIndex
```
 
The use of a single file heterogenous library requires that the toolchain properly unbundle this archive 
during the link phase.
This unbundle-archive operation can be optimized by using a new tool to unbundle the heterogenous archive 
into a multip-file library. 
This demo will be enhanced to show how this unbundle-archive tool is used.
