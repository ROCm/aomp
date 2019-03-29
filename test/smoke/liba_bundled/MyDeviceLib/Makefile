#===--------- libm/Makefile -----------------------------------------------===
#
#                The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===
#
# Makefile:  Makefile to build static device library as archive of bc files
#            written by Greg Rodgers
#

LIB       = MyDeviceLib
LIBSRC1   = func_1v
LIBSRC2   = func_2v
LIBSRC3   = func_3v

# --- Standard Makefile check for AOMP installation ---
ifeq ("$(wildcard $(AOMP))","")
  ifneq ($(AOMP),)
    $(warning AOMP not found at $(AOMP))
  endif
  AOMP = $(HOME)/rocm/aomp
  ifeq ("$(wildcard $(AOMP))","")
    $(warning AOMP not found at $(AOMP))
    AOMP = /opt/rocm/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
AOMP_GPU ?= gfx900
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
MEM_PREFIX_1 = $(TMPDIR)/$(LIBSRC1)-$(ARCH)-$(AOMP_GPU)
MEM_PREFIX_2 = $(TMPDIR)/$(LIBSRC2)-$(ARCH)-$(AOMP_GPU)
MEM_PREFIX_3 = $(TMPDIR)/$(LIBSRC3)-$(ARCH)-$(AOMP_GPU)

CFLAGS  = -c -target $(HOST_TARGET) -fopenmp -fopenmp-targets=$(TRIPLE) \
            -Xopenmp-target=$(TRIPLE) \
            -march=$(AOMP_GPU) -O2

$(MEM_PREFIX_1).bundled.bc: $(LIBSRC1).c
	$(CC) $(CFLAGS) -emit-llvm $^ -o $@
$(MEM_PREFIX_2).bundled.bc: $(LIBSRC2).c
	$(CC) $(CFLAGS) -emit-llvm $^ -o $@
$(MEM_PREFIX_3).bundled.bc: $(LIBSRC3).c
	$(CC) $(CFLAGS) -emit-llvm $^ -o $@

lib$(LIB)-$(ARCH)-$(AOMP_GPU).a : $(MEM_PREFIX_1).bc $(MEM_PREFIX_2).bc $(MEM_PREFIX_3).bc
	$(AOMP)/bin/llvm-ar r $@ $^ 

$(MEM_PREFIX_1).bc: $(MEM_PREFIX_1).bundled.bc
	$(AOMP)/bin/clang-offload-bundler -unbundle -type bc -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$@,$(TMPDIR)/$(LIBSRC1).host.bc
$(MEM_PREFIX_2).bc: $(MEM_PREFIX_2).bundled.bc
	$(AOMP)/bin/clang-offload-bundler -unbundle -type bc -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$@,$(TMPDIR)/$(LIBSRC2).host.bc
$(MEM_PREFIX_3).bc: $(MEM_PREFIX_3).bundled.bc
	$(AOMP)/bin/clang-offload-bundler -unbundle -type bc -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$@,$(TMPDIR)/$(LIBSRC3).host.bc

#   ---------------------  create host library ------------------- 
#
lib$(LIB).a : $(TMPDIR)/$(LIBSRC1).o $(TMPDIR)/$(LIBSRC2).o $(TMPDIR)/$(LIBSRC3).o
	$(AOMP)/bin/llvm-ar r $@ $^ 

$(MEM_PREFIX_1).bundled.o : $(LIBSRC1).c
	$(CC) $(CFLAGS) $^ -o $@
$(MEM_PREFIX_2).bundled.o: $(LIBSRC2).c
	$(CC) $(CFLAGS) $^ -o $@
$(MEM_PREFIX_3).bundled.o: $(LIBSRC3).c
	$(CC) $(CFLAGS)  $^ -o $@

$(TMPDIR)/$(LIBSRC1).o: $(MEM_PREFIX_1).bundled.o
	$(AOMP)/bin/clang-offload-bundler -unbundle -type o -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$(MEM_PREFIX_1).device.bc,$@
$(TMPDIR)/$(LIBSRC2).o: $(MEM_PREFIX_2).bundled.o
	$(AOMP)/bin/clang-offload-bundler -unbundle -type o -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$(MEM_PREFIX_2).device.bc,$@
$(TMPDIR)/$(LIBSRC3).o: $(MEM_PREFIX_3).bundled.o
	$(AOMP)/bin/clang-offload-bundler -unbundle -type o -inputs=$^ -targets=openmp-$(TRIPLE)-$(AOMP_GPU),host-$(HOST_TARGET) -outputs=$(MEM_PREFIX_3).device.bc,$@

clean:
	rm -f $(TMPDIR)/*.bc $(TMPDIR)/*.ll $(TMPDIR)/*.o *.a
	rmdir $(TMPDIR)

$(shell mkdir -p $(TMPDIR))
