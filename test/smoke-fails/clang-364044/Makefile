include ../../Makefile.defs

TESTNAME     = clang-364044
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-364044.cc
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -O3 -v -c 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
