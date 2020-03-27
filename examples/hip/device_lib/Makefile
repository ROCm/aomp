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

writeIndex: writeIndex.cpp lib_mylib.a lib_mylib-amdgcn.bc
	@echo
	@echo " 4. Compile app with HIP using _mylib SDL"
	$(HIPCC) -L$(PWD) -l_mylib writeIndex.cpp -o $@

#  Build Host Static Library 
inc_arrayval.o: inc_arrayval.c
	@echo
	@echo " 1. Compile function source to create host object $@"
	$(CLANG) -c $^ -o $@
lib_mylib.a: inc_arrayval.o
	@echo
	@echo " 2. Use ar command to create the host static library $@"
	ar rcs $@ $^ 

#  Build Device Static Library as a _mylib.bc file
lib_mylib-amdgcn.bc: inc_arrayval.c
	@echo
	@echo " 3. Compile same function source to create static device library (SDL)"
	$(CLANG) -c -emit-llvm -target amdgcn-amd-amdhsa $^ -o $@


run: writeIndex
	./writeIndex

clean:
	@[ -f ./writeIndex ] && rm ./writeIndex ; true
	@[ -f ./inc_arrayval.o ] && rm inc_arrayval.o ; true
	@[ -f ./lib_mylib.a ] && rm lib_mylib.a ; true
	@[ -f ./lib_mylib-amdgcn.bc ] && rm lib_mylib-amdgcn.bc ; true
