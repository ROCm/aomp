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
    AOMP = /usr/lib/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
AOMP_GPU ?= gfx900
CC        = $(AOMP)/bin/clang
UNAMEP    = $(shell uname -m)
HOST_TARGET = $(UNAMEP)-pc-linux-gnu

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  TRIPLE  = nvptx64-nvidia-cuda
  ARCH    = nvptx
else
  TRIPLE  = amdgcn-amd-amdhsa
  ARCH    = amdgcn
endif

TMPDIR   ?= ./build
CFLAGS  = -c -target $(HOST_TARGET) -fopenmp -fopenmp-targets=$(TRIPLE) \
            -Xopenmp-target=$(TRIPLE) \
            -march=$(AOMP_GPU) -O2

#   ---------------------  create bundle library -------------------
#
lib$(LIB).a : $(TMPDIR)/$(LIBSRC1).o $(TMPDIR)/$(LIBSRC2).o $(TMPDIR)/$(LIBSRC3).o
	ar r $@ $^

$(TMPDIR)/$(LIBSRC1).o: $(LIBSRC1).c
	$(CC) $(CFLAGS) $^ -o $@
$(TMPDIR)/$(LIBSRC2).o: $(LIBSRC2).c
	$(CC) $(CFLAGS) $^ -o $@
$(TMPDIR)/$(LIBSRC3).o: $(LIBSRC3).c
	$(CC) $(CFLAGS)  $^ -o $@

clean:
	rm -f $(TMPDIR)/*.bc $(TMPDIR)/*.ll $(TMPDIR)/*.o *.a
	rmdir $(TMPDIR)

$(shell mkdir -p $(TMPDIR))
