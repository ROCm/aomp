include ../../Makefile.defs

TESTNAME     = target-in-other-source
TESTSRC_MAIN = vmul.c
TESTSRC_AUX  = target.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
