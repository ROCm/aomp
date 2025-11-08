include ../../Makefile.defs

TESTNAME     = clang-ifaces
TESTSRC_MAIN = clang-ifaces.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -O0
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
