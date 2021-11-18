#-----------------------------------------------------------------------
#  ROCM_PATH:       Defaults to /opt/rocm
#  ROCM_GPU:        Auto detected via rocm_agent_enumerator
#  ROCM_GPUTARGET:  Defaults to amdgcn-amd-amdhsa
#
#  make run
#
#  Run "make help" to see other options for this Makefile
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

TESTNAME =hello
FILETYPE =f90

ROCM_PATH ?= /opt/rocm
ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

INSTALLED_GPU  = $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
ROCM_GPU       ?= $(INSTALLED_GPU)
CC              = $(ROCM_PATH)/llvm/bin/clang

ifeq ($(TARGETS),)
TARGETS =-march=$(ROCM_GPU)
endif

FORT   =$(ROCM_PATH)/llvm/bin/flang
LFLAGS = 
FFLAGS =-fopenmp -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET)

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(FORT) $(FFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(FORT) -c $(FFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o 
	$(FORT) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTNAME).$(FILETYPE)
	$(FORT) -c -S -emit-llvm $(FFLAGS) $(TARGETS) $^

$(TESTNAME).s: $(TESTNAME).$(FILETYPE)
	$(FORT)  -c -S $(FFLAGS) $(TARGETS)  $^

help: 
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTNAME).$(FILETYPE)"
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
	@echo "  ROCM_PATH default: /opt/rocm                                 value: $(ROCM_PATH)"
	@echo "  ROCM_GPU  default: auto detected by rocm_agent_enumerator    value: $(ROCM_GPU)"
	@echo "  TARGETS   default: --offload-arch=$(ROCM_GPU)"
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
