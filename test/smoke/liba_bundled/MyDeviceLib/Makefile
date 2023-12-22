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
include ../../../Makefile.defs

LIB       = MyDeviceLib
LIBSRC1   = func_1v
LIBSRC2   = func_2v
LIBSRC3   = func_3v

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  TRIPLE  = nvptx64-nvidia-cuda
  ARCH    = nvptx
else
  TRIPLE  = amdgcn-amd-amdhsa
  ARCH    = amdgcn
endif

TMPDIR    = ./build
CFLAGS  = -c $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=$(TRIPLE) \
            -Xopenmp-target=$(TRIPLE) \
						-march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES) -O2

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
