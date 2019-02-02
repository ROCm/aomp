#===--------- libm/Makefile -----------------------------------------------===
#
#                The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===
#
# Makefile:  Makefile for building a single Device BC Library (DBCL)
#            written by Jan Sjodin and Greg Rodgers
#

LIB       = libm
LIBMSRC   = libm.c
LIBMPPSRC = libm.cpp

AOMP     ?= /opt/rocm/aomp
AOMP_GPU ?= sm_60
CXX       = $(AOMP)/bin/clang++
CC        = $(AOMP)/bin/clang
UNAMEP    = $(shell uname -p)
HOST_TARGET = $(UNAMEP)-pc-linux-gnu

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  TRIPLE  = nvptx64-nvidia-cuda
  ARCH    = nvptx
else
  TRIPLE  = amdgcn-amd-amdhsa
  ARCH    = amdgcn
endif

TMPDIR   ?= ./build
# Example, to build DBCL in the AOMP installation, do this
# LIBPATH=$AOMP/lib make
LIBPATH  ?= $(TMPDIR)

TMPNAME   = $(TMPDIR)/$(LIB)-$(ARCH)-$(AOMP_GPU)
LIBNAME   = $(LIBPATH)/libdevice/$(LIB)-$(ARCH)-$(AOMP_GPU)

CXXFLAGS  = -c -target $(HOST_TARGET) -fopenmp -fopenmp-targets=$(TRIPLE) \
            -Xopenmp-target=$(TRIPLE) \
            -march=$(AOMP_GPU) -emit-llvm -O1
CFLAGS    = $(CXXFLAGS)

CXXFLAGS += $(EXTRA_CXXFLAGS)
CFLAGS   += $(EXTRA_CFLAGS)

$(TMPNAME).ll : $(LIBNAME).bc
	$(AOMP)/bin/llvm-dis $^ -o $@

$(TMPNAME).b.cpp.bc: $(LIBMPPSRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TMPNAME).b.c.bc: $(LIBMSRC)
	$(CC) $(CFLAGS) $^ -o $@

$(TMPNAME).cpp.bc: $(TMPNAME).b.cpp.bc
	$(AOMP)/bin/clang-offload-bundler -unbundle -type bc -inputs=$^ -targets=openmp-$(TRIPLE),host-$(HOST_TARGET) -outputs=$@,$(TMPNAME).host.cpp.bc

$(TMPNAME).c.bc: $(TMPNAME).b.c.bc
	$(AOMP)/bin/clang-offload-bundler -unbundle -type bc -inputs=$^ -targets=openmp-$(TRIPLE),host-$(HOST_TARGET) -outputs=$@,$(TMPNAME).host.c.bc

$(LIBNAME).bc: $(TMPNAME).c.bc $(TMPNAME).cpp.bc
	mkdir -p $(LIBPATH)/libdevice
	$(AOMP)/bin/llvm-link $^ -o $@

clean:
	rm -f $(TMPDIR)/*.bc $(TMPDIR)/*.ll

clean-out: clean
	rm -rf $(LIBPATH)/libdevice

$(shell mkdir -p $(TMPDIR))
