#-----------------------------------------------------------------------
#
#  Makefile: demo Makefile for Compiler example simple_offload
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

# All Compile examples Makefiles should include find_gpu_and_install_dir.mk
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

TESTNAME := simple_offload
FILETYPE := f95
TESTSRC := $(TESTNAME).$(FILETYPE)

# If out-of-tree, prefix all inputs with $(mkfile_dir)
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

FORT   := $(LLVM_INSTALL_DIR)/bin/$(FLANG)
FFLAGS := -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)
LFLAGS := $(FFLAGS) 
ifneq ($(LLVM_COMPILER_NAME),clang)
  FORTENV := LIBOMPTARGET_KERNEL_TRACE=1
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRC)
	$(FORT) $(FFLAGS) $^ -o $@

run: $(TESTNAME)
	$(FORTENV) ./$(TESTNAME)

include $(mkfile_dir)../../inc/obin_fortran.mk

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTSRC)
	$(FORT) -c -S -emit-llvm $(FFLAGS) $^

$(TESTNAME).s: $(TESTSRC)
	$(FORT)  -c -S $(FFLAGS) $^


include $(mkfile_dir)../../inc/help_fortran.mk

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
