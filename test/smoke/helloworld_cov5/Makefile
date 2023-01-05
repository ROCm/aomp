include ../../Makefile.defs

TESTNAME     = helloworld_cov5
TESTSRC_MAIN = helloworld_cov5.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -mcode-object-version=5
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
