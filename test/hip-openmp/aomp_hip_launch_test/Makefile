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

TESTNAME =hiplaunch
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

CC = $(AOMP)/bin/clang
AOMPHIP ?= $(AOMP)
DEVICE_LIB_PATH=$(AOMPHIP)/amdgcn/bitcode/

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

HIPFLAGS = -x hip $(PLATFORM) -O2 --offload-arch=$(AOMP_GPU) -lpthread -std=c++11 -lamdhip64
OMP_TARGET_FLAGS = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU) --save-temps 
CC=$(AOMP)/bin/clang

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME).exe: $(TESTNAME).$(FILETYPE)
	$(CC) $(OMP_TARGET_FLAGS) test_omp.c -o test_omp.exe
	test -f test_omp-host-x86_64-unknown-linux-gnu.o
	test -f a.out-openmp-amdgcn-amd-amdhsa-$(AOMP_GPU)
	DEVICE_LIB_PATH=$(DEVICE_LIB_PATH) $(AOMPHIP)/bin/hipcc -o hiplaunch.exe hiplaunch.cpp -fopenmp
	#/opt/rocm/bin/hipcc -o hiplaunch.exe hiplaunch.c -fopenmp # this needs code-obj-v3 fix , ROCm 3.0 ???
#	/work/hip-clang-bld/HIP/bin/hipcc -o hiplaunch.exe hiplaunch.c -fopenmp # this needs code-obj-v3 fix , ROCm 3.0 ???

run: $(TESTNAME).exe
	$(eval KERNEL_NAME=$(shell nm test_omp-host-x86_64-unknown-linux-gnu.o | grep ' __omp_offloading' | sed 's/[0-9a-f]\+ t //'))
	echo KERNEL_NAME=$(KERNEL_NAME)
	echo AOMP_GPU=$(AOMP_GPU)
	LIBOMPTARGET_KERNEL_TRACE=1 ./test_omp.exe
	./$(TESTNAME).exe $(KERNEL_NAME) $(AOMP_GPU)

# Cleanup anything this makefile can create
clean:
	rm -f *.bc *.i *.o *.s test_omp.lk test_omp.exe hiplaunch.exe a.out-*
