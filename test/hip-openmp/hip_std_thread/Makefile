#-----------------------------------------------------------------------
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn targets begin with gfx. nvptx targets begin with sm_
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

TESTNAME =bit_extract
FILETYPE =cpp

UNAMEP = $(shell uname -p)
AOMP_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
  AOMP_CPUTARGET = ppc64le-linux-gnu
endif
ifeq ($(AOMP),)
# --- Standard Makefile check for AOMP installation ---
ifeq ("$(wildcard $(AOMP))","")
  ifneq ($(AOMP),)
    $(warning AOMP not found at $(AOMP))
  endif
  AOMP = $(HOME)/rocm/aomp
  ifeq ("$(wildcard $(AOMP))","")
    $(warning AOMP not found at $(AOMP))
    AOMP = /usr/lib/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
endif
ifeq ($(CUDA),)
CUDA = /usr/local/cuda
endif
ifeq ($(AOMP_GPU),)
AOMP_GPU = $(shell $(AOMP)/bin/mygpu -d gfx900) #get installed card, if unknown return gfx900
endif

CC = $(AOMP)/bin/clang++ -std=c++11

# Add cudart only if we have an Nvidia sm_ target
# We have not tested CUDA+OpenMP target offload
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  LFLAGS = -L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib
  TRIPLE = nvptx64-nvidia-cuda
  PLATFORM = -D__HIP_PLATFORM_NVCC__=1
else
  TRIPLE = amdgcn-amd-amdhsa
  PLATFORM = -D__HIP_PLATFORM_HCC__=1
endif

#  These HIPFLAGS provide HIP+OpenMP for CPU only. No target offload
AOMPHIP ?= $(AOMP)
HIPLIBS = -L $(AOMPHIP)/hip -L $(AOMPHIP)/lib
HIPFLAGS = -x hip $(PLATFORM) -O2 --offload-arch=$(AOMP_GPU) -lpthread $(HIPLIBS) -lamdhip64  -Wl,-rpath,$(AOMPHIP)/lib

#  These OMP_TARGET_FLAGS added to HIPFLAGS provide HIP+OpenMP with GPU target offload
OMP_TARGET_FLAGS = 

CFLAGS = $(HIPFLAGS) $(OMP_TARGET_FLAGS)

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $^

obin:	$(TESTNAME).o 
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin

help: 
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTNAME).$(FILETYPE)"
	@echo "  Compiler: 		$(CC)"
	@echo "  Compiler flags: 	$(CFLAGS)"
	@echo
	@echo "Avalable Targets:"
	@echo "  make			// build binary $(TESTNAME)"
	@echo "  make run		// run $(TESTNAME)"
	@echo "  make $(TESTNAME).o	// compile, be, & assemble : -c"
	@echo "  make obin         	// link step only"
	@echo "  make run_obin     	// run obin "
	@echo "  make clean		// cleanup files"
	@echo "  make help 		// this help"
	@echo
	@echo "Environment Variables:"
	@echo "  AOMP      default: $(HOME)/rocm/aomp 	value: $(AOMP)"
	@echo "  AOMP_GPU  default: gfx900                	value: $(AOMP_GPU)"
	@echo

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
