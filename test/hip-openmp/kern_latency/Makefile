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

TESTNAME =kern_latency
FILETYPE =cpp

ifeq ($(TARGETS),)
TARGETS =--offload-arch=$(AOMP_GPU)$(AOMP_TARGET_FEATURES)
endif

CC       =$(AOMP)/bin/clang++ -std=c++11
#CFLAGS   =-O3 $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target amdgcn-amd-amdhsa #--offload-arch=$(AOMP_GPU)
#CFLAGS   =-x hip -O3 $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900
CFLAGS   =-x hip -O3 $(AOMP_CPUTARGET) -fopenmp
#CFLAGS   =-O3 -fopenmp $(AOMP_CPUTARGET)

# Add cudart only if we have an Nvidia sm_ target
ifeq (sm_,$(findstring sm_,$(TARGETS)))
  LFLAGS +=-L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib
  CFLAGS +=-x cuda -I$(CUDA)/include
else
  HIPLIBS = -L $(AOMPHIP)/hip -L $(AOMPHIP)/lib
  VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
  ifeq ($(shell expr $(VERS) \>= 12.0), 1)
    RPTH = -Wl,-rpath,$(AOMPHIP)/lib
  endif
  CFLAGS += -x hip $(HIPLIBS) -lamdhip64 $(RPTH)
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME) 2>&1 | ./timestampit.sh

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o 
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTNAME).$(FILETYPE)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGETS) $^

$(TESTNAME).s: $(TESTNAME).$(FILETYPE)
	$(CC)  -c -S $(CFLAGS) $(TARGETS)  $^

include ../Makefile.HelpClean 
