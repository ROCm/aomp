include ../../Makefile.defs

TESTNAME     = data_share1
TESTSRC_MAIN = data_share1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED  = ASAN_RUNTIME

include ../Makefile.rules
