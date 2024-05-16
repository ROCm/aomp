include ../../Makefile.defs

TESTNAME     = helloworld_cov4
TESTSRC_MAIN = helloworld_cov4.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -mcode-object-version=4
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
