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

TESTNAME =matrixmul
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
HIPFLAGS = -x hip $(PLATFORM) -O2 --offload-arch=$(AOMP_GPU) -target $(AOMP_CPUTARGET)  $(HIPLIBS) -lhip_hcc
#  These OMP_TARGET_FLAGS added to HIPFLAGS provide HIP+OpenMP with GPU target offload
OMP_TARGET_FLAGS = -fopenmp-targets=$(TRIPLE) -Xopenmp-target=$(TRIPLE) -march=$(AOMP_GPU) -target $(AOMP_CPUTARGET)

CFLAGS = -fopenmp -O2 

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): matsup.o hiptarg.o omptarg.o $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(LFLAGS) $^ -o $@ $(HIPFLAGS) $(OMP_TARGET_FLAGS) $(HIPLIBS) -lhip_hcc

run: $(TESTNAME)
	timeout 10 ./$(TESTNAME)

matsup.o:
	$(CC) $(CFLAGS) -o $@ matsup.cpp -c

hiptarg.o:
	$(CC) $(HIPFLAGS) $(CFLAGS) -o $@ hiptarg.cpp -c

omptarg.o:
	$(CC) $(OMP_TARGET_FLAGS) $(CFLAGS) -o $@ omptarg.cpp -c

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: matsup.o hiptarg.o omptarg.o $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $^ 

obin:	$(TESTNAME).o 
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin


# Cleanup anything this makefile can create
clean::
	rm -f $(TESTNAME) $(TESTNAME).a llbin sbin obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.log
