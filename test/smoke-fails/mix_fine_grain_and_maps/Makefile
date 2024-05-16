include ../../Makefile.defs

TESTNAME     = simple_ptr
TESTSRC_MAIN = simple_ptr.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++ -DREQUIRES_USM
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
