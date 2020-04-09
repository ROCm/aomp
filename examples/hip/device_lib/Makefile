#-----------------------------------------------------------------------
#  Makefile: Demo how to build heterogeneous static device library
#-----------------------------------------------------------------------
# MIT License
# Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

AOMP ?= /usr/lib/aomp
HIPCC = $(AOMP)/bin/hipcc
OCLCC = $(AOMP)/bin/clang
CLANG = $(AOMP)/bin/clang
FLANG = $(AOMP)/bin/flang
LINKLL = $(AOMP)/bin/llvm-link

AMDTARGET = -target amdgcn-amd-amdhsa
OCLFLAGS  = -I$(PWD) -fcolor-diagnostics -Werror -Wno-error=atomic-alignment -x cl -Xclang -cl-std=CL2.0 $(AMDTARGET) -fvisibility=protected -fomit-frame-pointer -Xclang -finclude-default-header -nogpulib 
OMPFLAGS = -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=`$(AOMP)/bin/mygpu`

writeIndex: writeIndex.hip lib_mylib.a lib_myomplib.a lib_mylib-amdgcn.bc
	@echo
	@echo " 4. Compile HIP application that uses both the _mylib and _myomplib  SDLs"
	$(HIPCC) -L$(PWD) -l_myomplib -l_mylib writeIndex.hip -o $@

#  Build Host Static Library 
inc_arrayval.o: inc_arrayval.c
	@echo
	@echo " 1.1 Compile C library function source to create host object $@"
	$(CLANG) -c $^ -o $@
dec_arrayval.o: dec_arrayval.f95
	@echo
	@echo " 1.2 Compile f95 library function source to create host object $@"
	$(FLANG) -c $^ -o $@
lib_mylib.a: inc_arrayval.o dec_arrayval.o
	@echo
	@echo " 1.3 Use Linux ar command to create the host static library $@"
	ar rcs $@ $^ 

#  Build amdgcn Static Device library (SDL)
#AMDTARGET = -target nvptx64-nvidia-cuda
inc_arrayval-amdgcn.bc: inc_arrayval.c
	@echo
	@echo " 3.1 Compile c library source again to create device bitcode"
	$(CLANG) -c -emit-llvm $(AMDTARGET) $^ -o $@
dec_arrayval-amdgcn.bc: dec_arrayval.cl
	@echo
	@echo " 3.2 Compile OpenCL library source again to create device bitcode"
	$(OCLCC) $(OCLFLAGS) -c -emit-llvm $^ -o $@
# FORTRAN device only compile not working yet so just use OpenCL for now
#dec_arrayval-amdgcn.bc: dec_arrayval.f95
#	@echo
#	@echo " 2.2 Link bitcode files to create static device library (SDL)"
#	$(FLANG) -c -emit-llvm $(AMDTARGET) -I$(AOMP)/include $^ -o $@
lib_mylib-amdgcn.bc: inc_arrayval-amdgcn.bc dec_arrayval-amdgcn.bc
	@echo
	@echo " 3.3 Link bitcode files to create static device library (SDL)"
	$(LINKLL) $^ -o $@


lib_myomplib.a:  openmp_target_lib.o
	@echo
	@echo " 2.2 Create heterogeneous archive, a truly special capability"
	ar rcs $@ $^ 

openmp_target_lib.o:  openmp_target_lib.c
	@echo
	@echo " 2.1 Use OpenMP to compile heterogenous functions"
	$(CLANG) $(OMPFLAGS) -c $^ -o $@

run: writeIndex
	./writeIndex

clean:
	@[ -f ./writeIndex ] && rm ./writeIndex ; true
	@[ -f ./inc_arrayval.o ] && rm inc_arrayval.o ; true
	@[ -f ./dec_arrayval.o ] && rm dec_arrayval.o ; true
	@[ -f ./lib_mylib.a ] && rm lib_mylib.a ; true
	@[ -f ./lib_mylib-amdgcn.bc ] && rm lib_mylib-amdgcn.bc ; true
	@[ -f ./lib_myomplib.a ] && rm lib_myomplib.a ; true
	@[ -f ./inc_arrayval-amdgcn.bc ] && rm inc_arrayval-amdgcn.bc ; true
	@[ -f ./dec_arrayval-amdgcn.bc ] && rm dec_arrayval-amdgcn.bc ; true
	@[ -f ./openmp_target_lib.o ] && rm openmp_target_lib.o ; true
