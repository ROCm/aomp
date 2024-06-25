include ../../Makefile.defs

TESTNAME     = memory_mgmt_pinned_predefined
TESTSRC_MAIN = memory_mgmt_pinned_predefined.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules
