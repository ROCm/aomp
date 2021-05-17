include ../../Makefile.defs

TESTNAME     = alt_clang
TESTSRC_MAIN = alt_clang.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG) -famd-opt  -\#\#\#
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules


run: $(TESTNAME)
	echo pass

