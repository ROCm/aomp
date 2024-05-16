# AOMP -- Examples for stdpar

This directory contains these examples that use stdpar GPU acceleration.

- babelstream:  builds and executes two variants of babelstream.
- tsp:  build GPU accelerated traveling salesman problem

These examples require that rocthrust and rocprim be installed.
If LLVM_INSTALL_DIR specifies something other than the latest installed
ROCm compiler (found in /opt/rocm/lib/llvm), the Makefile for these
examples will still find rocthrust and rocprim in /opt/rocm.
For example, to test the aomp development compiler, set LLVM_INSTALL_DIR=/usr/lib/aomp.
