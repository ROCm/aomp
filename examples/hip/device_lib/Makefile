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
CLANG = $(AOMP)/bin/clang
FLANG = $(AOMP)/bin/flang
UNBUNDLE = $(AOMP)/bin/clang-offload-bundler

AOMPHIP ?= $(AOMP)
DEVICE_LIB_PATH = $(AOMPHIP)/amdgcn/bitcode

MYGPU = $(shell $(AOMPHIP)/bin/mygpu -d gfx900)
TRIPLE = amdgcn-amd-amdhsa
#OMPFLAGS = -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(MYGPU)
OMPFLAGS = -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(MYGPU)

writeIndex: writeIndex.hip lib_mylib.a libbc-_mylib-amdgcn.a   lib_myomplib.a 
	@echo
	@echo " ----------  3 Build HIP application using both SDLs ----------------" 
	@echo
	@echo " 3. Create HIP app using both _mylib and _myomplib SDLs. 		$@"
	DEVICE_LIB_PATH=$(DEVICE_LIB_PATH) $(HIPCC) -std=c++11 -fgpu-rdc -L$(PWD) -l_myomplib -l_mylib writeIndex.hip -o $@ -std=c++11

inc_arrayval.o: inc_arrayval.c
	@echo
	@echo " ----------  1 Build _mylib SDL (two files) ------------------------"
	@echo
	@echo " 1.1 Compile C library function source to create host-only object 	$@"
	$(CLANG) -c $^ -o $@
dec_arrayval.o: dec_arrayval.f95
	@echo
	@echo " 1.2 Compile FORTRAN function to create 2nd host-only object.		$@"
	$(FLANG) -c $^ -o $@
lib_mylib.a: inc_arrayval.o dec_arrayval.o
	@echo
	@echo " 1.3 Use Linux ar command to create the host-only archive file.		$@"
	ar rcs $@ $^ 
inc_arrayval-amdgcn.bc: inc_arrayval.c
	@echo
	@echo " 1.4 Re-Compile c library source to create a device bitcode.		$@"
	$(CLANG) -c -emit-llvm -mcpu=$(MYGPU) -target $(TRIPLE) -nogpulib $^ -o $@

dec_arrayval-amdgcn.bc: dec_arrayval.c
	@echo
	@echo " 1.5 Re-Compile c Code to create a 2nd device bitcode file.		$@"
	$(CLANG) -c -emit-llvm -mcpu=$(MYGPU) -target $(TRIPLE) -nogpulib $^ -o $@

libbc-_mylib-amdgcn.a: inc_arrayval-amdgcn.bc dec_arrayval-amdgcn.bc
	@echo
	@echo " 1.6 Use Linux ar command to create device-specific archive. 		$@"
	ar rcs $@ $^ 

_myomplib.o:  _myomplib.c
	@echo
	@echo " ----------  2 Build _myomplib SDL (single fat archive file) --------"
	@echo
	@echo " 2.1 Compile heterogenous functions into a bundled object. 		$@"
	$(CLANG) $(OMPFLAGS) -c $^ -o $@
lib_myomplib.a:  _myomplib.o
	@echo
	@echo " 2.2 Create heterogeneous archive from bundled objects.			$@"
	ar rcs $@ $^ 

run: writeIndex
	./writeIndex

clean:
	@[ -f ./writeIndex ] && rm ./writeIndex ; true
	@[ -f ./inc_arrayval.o ] && rm inc_arrayval.o ; true
	@[ -f ./dec_arrayval.o ] && rm dec_arrayval.o ; true
	@[ -f ./lib_mylib.a ] && rm lib_mylib.a ; true
	@[ -f ./libbc-_mylib-amdgcn.a ] && rm libbc-_mylib-amdgcn.a ; true
	@[ -f ./lib_myomplib.a ] && rm lib_myomplib.a ; true
	@[ -f ./inc_arrayval-amdgcn.bc ] && rm inc_arrayval-amdgcn.bc ; true
	@[ -f ./dec_arrayval-amdgcn.bc ] && rm dec_arrayval-amdgcn.bc ; true
	@[ -f ./_myomplib.o ] && rm _myomplib.o ; true
