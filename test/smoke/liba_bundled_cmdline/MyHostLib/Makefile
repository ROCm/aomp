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

LIB       = MyHostLib
LIBSRC1   = hfunc1
LIBSRC2   = hfunc2
LIBSRC3   = hfunc3

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
CC        = $(AOMP)/bin/clang
UNAMEP    = $(shell uname -p)
HOST_TARGET = $(UNAMEP)-pc-linux-gnu

TMPDIR   ?= ./build
CFLAGS  = -c -target $(HOST_TARGET) -O2

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
