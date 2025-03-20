#-----------------------------------------------------------------------
#
#  Makefile: demo Makefile for Compiler example fortran_hip_interop
#
#  Run "make help" to see how to use this Makefile
#
#-----------------------------------------------------------------------
# MIT License
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
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
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

TESTNAME   := fortran_hip_interop
HIPOBJECT  := fortran_callable_init
FILETYPE   := f95
TESTSRC    := $(TESTNAME).$(FILETYPE)
TESTSRCHIP := fortran_callable_init.hip

ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
  TESTSRCHIP := $(mkfile_dir)$(TESTSRCHIP)
endif

FORT       := $(LLVM_INSTALL_DIR)/bin/$(FLANG) -fPIC
FFLAGS     := -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)
LFLAGS     := $(FFLAGS)
LFLAGS     += -L$(HIPDIR)/lib -lamdhip64 -Wl,-rpath,$(HIPDIR)/lib
ifneq ($(LLVM_COMPILER_NAME),clang)
  FORTENV := LIBOMPTARGET_KERNEL_TRACE=1
endif

$(TESTNAME): $(TESTSRC) fortran_callable_init.o
	$(FORT) $(LFLAGS) $^ -o $@

fortran_callable_init.o : $(TESTSRCHIP)
	$(HIPCC) -c --offload-arch=$(LLVM_GPU_ARCH) -fPIC $^ -o $@

run: $(TESTNAME)
	$(FORTENV) ./$(TESTNAME)

help:
	@echo
	@echo "FORTRAN Source:		$(TESTSRC)"
	@echo "HIPSource:		$(TESTSRCHIP)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Compiler install dir:	$(LLVM_INSTALL_DIR)"
	@echo "Compiler name:		$(LLVM_COMPILER_NAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Target triple:		$(LLVM_GPU_TRIPLE)"
	@echo "Compiler: 		$(FORT)"
	@echo "Compile flags:		$(FFLAGS)"
	@echo "Linkflags:		$(LFLAGS)"
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_INSTALL_DIR     LLVM installation directory"
	@echo "  LLVM_GPU_ARCH        Target GPU, e.g. sm_30, gfx90a"
	@echo


# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
	@[ -f ./$(HIPOBJECT).o ] && rm $(HIPOBJECT).o ; true
