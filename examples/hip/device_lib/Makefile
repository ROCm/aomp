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

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir :=
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CLANG = $(LLVM_INSTALL_DIR)/bin/clang
FLANG ?= $(LLVM_INSTALL_DIR)/bin/flang
ifeq ("$(wildcard $(FLANG))","")
  FLANG = $(LLVM_INSTALL_DIR)/bin/flang-new
endif
BUNDLER = $(LLVM_INSTALL_DIR)/bin/clang-offload-bundler

DEVICE_LIB_PATH = $(HIPDIR)/amdgcn/bitcode

OMPFLAGS = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

writeIndex: $(_in_dir)writeIndex.hip lib_mylib.a bundled/lib_myomplib.a
	@echo
	@echo " ----------  3 Build HIP application using both SDLs ----------------"
	@echo
	@echo " 3.1 Create HIP application using both _mylib and _myomplib SDLs.        $@"
	DEVICE_LIB_PATH=$(DEVICE_LIB_PATH) $(HIPCC) -std=c++11 -fgpu-rdc -L$(PWD)/bundled -l_myomplib -L$(PWD) -l_mylib $(_in_dir)writeIndex.hip -o $@ -std=c++11

inc_arrayval.o: $(_in_dir)inc_arrayval.c
	@echo
	@echo " 1.1 Compile C library function source to create host-only object        $@"
	$(CLANG) -c $^ -o $@

dec_arrayval.o: $(_in_dir)dec_arrayval.f95
	@echo
	@echo " 1.2 Compile FORTRAN function to create 2nd host-only object . $@"
	$(FLANG) -c $^ -o $@

inc_arrayval-amdgcn.bc: $(_in_dir)inc_arrayval.c 
	@echo
	@echo " 1.3 Re-Compile c library source to create a device bitcode with -fopenmp-host-ir-file-path host.bc  $@"
	$(CLANG) -c -emit-llvm -mcpu=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES) -target $(LLVM_GPU_TRIPLE) -nogpulib $^ -o $@

dec_arrayval-amdgcn.bc: $(_in_dir)dec_arrayval.c
	@echo
	@echo " 1.4 Re-Compile c Code to create a 2nd device bitcode file.		$@"
	$(CLANG) -c -emit-llvm -mcpu=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES) -target $(LLVM_GPU_TRIPLE) -nogpulib $^ -o $@

inc_arrayval-amdgcn.o: inc_arrayval-amdgcn.bc inc_arrayval.o
	@echo
	@echo " 1.5 create $@ from $^ using bundler. "
	$(BUNDLER) -type=o -targets=openmp-$(LLVM_GPU_TRIPLE)-$(LLVM_GPU_ARCH),host-x86_64-unknown-linux-gnu -output=$@ -input=inc_arrayval-amdgcn.bc -input=inc_arrayval.o

dec_arrayval-amdgcn.o: dec_arrayval-amdgcn.bc dec_arrayval.o
	@echo
	@echo " 1.6 create $@ from $^ using bundler."
	$(BUNDLER) -type=o -targets=openmp-$(LLVM_GPU_TRIPLE)-$(LLVM_GPU_ARCH),host-x86_64-unknown-linux-gnu -output=$@ -input=dec_arrayval-amdgcn.bc -input=dec_arrayval.o

lib_mylib.a: inc_arrayval-amdgcn.o dec_arrayval-amdgcn.o
	@echo
	@echo " 1.7 Use llvm-ar to create archive $@ from $^"
	$(LLVM_INSTALL_DIR)/bin/llvm-ar rcs $@ $^

_myomplib.o:  $(_in_dir)_myomplib.c
	@echo
	@echo " ----------  2 Build _myomplib SDL (single fat archive file) --------"
	@echo
	@echo " 2.1 Compile heterogenous functions into a packaged object. 		$@"
	$(CLANG) $(OMPFLAGS) -c $^ -o $@
lib_myomplib.a:  _myomplib.o
	@echo
	@echo " 2.2 Create heterogeneous archive lib from packaged object.			$@"
	$(LLVM_INSTALL_DIR)/bin/llvm-ar rcs $@ $^

bundled/lib_myomplib.a: lib_myomplib.a
	@echo
	@echo " 2.3 Create bundled archive lib from packaged archive lib.			$@"
	mkdir -p bundled ; $(mkfile_dir)../../rebundle_hip_lib.sh $(PWD)/$^ $(PWD)/$@

run: writeIndex
	./writeIndex

clean:
	@[ -f ./writeIndex ] && rm ./writeIndex ; true
	@[ -f ./inc_arrayval.o ] && rm inc_arrayval.o ; true
	@[ -f ./dec_arrayval.o ] && rm dec_arrayval.o ; true
	@[ -f ./lib_mylib.a ] && rm lib_mylib.a ; true
	@[ -f ./lib_myomplib.a ] && rm lib_myomplib.a ; true
	@[ -f ./bundled/lib_myomplib.a ] && rm bundled/lib_myomplib.a ; true
	@[ -f ./inc_arrayval-amdgcn.bc ] && rm inc_arrayval-amdgcn.bc ; true
	@[ -f ./inc_arrayval-amdgcn.o ] && rm inc_arrayval-amdgcn.o ; true
	@[ -f ./dec_arrayval-amdgcn.bc ] && rm dec_arrayval-amdgcn.bc ; true
	@[ -f ./dec_arrayval-amdgcn.o ] && rm dec_arrayval-amdgcn.o ; true
	@[ -f ./_myomplib.o ] && rm _myomplib.o ; true
