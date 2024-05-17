include ../../Makefile.defs

TESTNAME     = alt_veclib
TESTSRC_MAIN = alt_veclib.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG) -fveclib=AMDLIBM  # -\#\#\#
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules


#run: $(TESTNAME)
#	echo pass

