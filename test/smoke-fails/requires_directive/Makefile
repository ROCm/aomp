include ../../Makefile.defs

TESTNAME     = requires_directive
TESTSRC_MAIN = requires_directive.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
