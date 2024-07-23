NOOPT        = 1
include ../../Makefile.defs
CFLAGS      += -O3

TESTNAME     = jacobi
TESTSRC_MAIN = jacobi.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNCMD       = ./$(TESTNAME) 1024 10

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
