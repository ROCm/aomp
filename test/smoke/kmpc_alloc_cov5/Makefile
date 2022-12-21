include ../../Makefile.defs

TESTNAME     = kmpc_alloc_cov5
TESTSRC_MAIN = kmpc_alloc_cov5.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -mcode-object-version=5

include ../Makefile.rules
