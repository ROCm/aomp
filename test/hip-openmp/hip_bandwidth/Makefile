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

include ../../Makefile.defs
TESTNAME =bit_extract
FILETYPE =cpp

CC = $(AOMP)/bin/clang++

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
HIPLIBS = -L $(AOMPHIP)/hip -L $(AOMPHIP)/lib
VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
ifeq ($(shell expr $(VERS) \>= 12.0), 1)
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib
endif
HIPFLAGS = -x hip $(PLATFORM) -O2 --offload-arch=$(AOMP_GPU)$(AOMP_TARGET_FEATURES) -lpthread -std=c++11  $(HIPLIBS) -lamdhip64  $(RPTH)

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

include ../Makefile.HelpClean
