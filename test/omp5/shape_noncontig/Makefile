include ../../Makefile.defs

TESTNAME     = shape_noncontig
TESTSRC_MAIN = shape_noncontig.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
