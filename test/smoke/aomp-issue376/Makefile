include ../../Makefile.defs

TESTNAME     = aomp-issue376
TESTSRC_MAIN = aomp-issue376.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD      = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
