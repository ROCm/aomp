include ../../Makefile.defs

TESTNAME     = helloworld
TESTSRC_MAIN = helloworld.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# The %n printf format does not work with libc printf so force emissary printf
ifeq ($(shell echo 'int main() { return 0; }' | $(CC) -fuse-emissary-print -xc - 2>/dev/null && echo true), true)
  EXTRA_CFLAGS=-fuse-emissary-print
endif

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
