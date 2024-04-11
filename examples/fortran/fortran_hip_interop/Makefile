#-----------------------------------------------------------------------
#
#  Makefile: demo Makefile for amdgcn target (for flang).
#            amdgcn targets begin with gfx. 
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

TESTNAME =fortran_hip_interop
HIPOBJECT =fortran_callable_init
FILETYPE =f95
TESTSRC = $(TESTNAME).$(FILETYPE)
TESTSRCHIP = fortran_callable_init.hip

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
  TESTSRCHIP := $(mkfile_dir)$(TESTSRCHIP)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

ifeq ($(TARGETS),)
TARGETS =--offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
endif

FORT   =$(LLVM_INSTALL_DIR)/bin/flang -fPIC -O2
LFLAGS = -L$(HIPDIR)/lib -lamdhip64 -Wl,-rpath,$(HIPDIR)/lib
FFLAGS = -fopenmp

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRC) fortran_callable_init.o
	$(FORT) $(FFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

fortran_callable_init.o : $(TESTSRCHIP)
	$(HIPCC) -c --offload-arch=$(LLVM_GPU_ARCH) -fPIC $^ -o $@

run: .FORCE $(TESTNAME)
	LIBOMPTARGET_KERNEL_TRACE=1 ./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTSRC)
	$(FORT) -c $(FFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o fortran_callable_init.o 
	$(FORT) $(FFLAGS) $(TARGETS) $(LFLAGS) $^ -o obin

run_obin: obin 
	LIBOMPTARGET_KERNEL_TRACE=1 ./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTSRC) fortran_callable_init.ll
	$(FORT) -c -S -emit-llvm $(FFLAGS) $(TARGETS) $(TESTSRC)

fortran_callable_init.ll : $(TESTSRCHIP)
	$(LLVM_INSTALL_DIR)/bin/hipcc -c -S -emit-llvm --offload-arch=$(LLVM_GPU_ARCH) $^

$(TESTNAME).s: $(TESTSRC) fortran_callable_init.s
	$(FORT)  -c -S $(FFLAGS) $(TARGETS) $(TESTSRC)

fortran_callable_init.s : $(TESTSRCHIP)
	$(LLVM_INSTALL_DIR)/bin/hipcc -c -S --offload-arch=$(LLVM_GPU_ARCH) $^

.FORCE:
	rm -f $(TESTNAME)

help: 
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTSRC)"
	@echo "  Compiler: 		$(FORT)"
	@echo "  Compiler flags: 	$(FFLAGS)"
	@echo
	@echo "Avalable Targets:"
	@echo "  make			// build binary $(TESTNAME)"
	@echo "  make run		// run $(TESTNAME)"
	@echo "  make $(TESTNAME).o	// compile, be, & assemble : -c"
	@echo "  make obin         	// link step only"
	@echo "  make run_obin     	// run obin "
	@echo "  make $(TESTNAME).s	// compile & backend steps : -c -S"
	@echo "  make $(TESTNAME).ll	// compile step only       : -c -S -emit-llvm"
	@echo "  make clean		// cleanup files"
	@echo "  make help 		// this help"
	@echo
	@echo "Environment Variables:"
	@echo "  LLVM_INSTALL_DIR  value: $(LLVM_INSTALL_DIR)"
	@echo "  LLVM_GPU_ARCH     value: $(LLVM_GPU_ARCH)"
	@echo "  LLVM_GPU_TRIPLE   value: $(LLVM_GPU_TRIPLE)"
	@echo "  CUDA      default: /usr/local/cuda       value: $(CUDA)"
	@echo "  TARGETS   default: --offload-arch=$(LLVM_GPU_ARCH)"
	@echo "              value: $(TARGETS)"
	@echo
	@echo "Link Flags:"
	@echo "  Link flags: $(LFLAGS)"
	@echo

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
	@[ -f ./$(HIPOBJECT).o ] && rm $(HIPOBJECT).o ; true
