include ../../Makefile.defs

TESTNAME     = lib-debug
TESTSRC_MAIN = lib-debug.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang -fopenmp-runtimelib=lib-debug
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules

RUNCMD       = ./$(TESTNAME) 2>&1 | grep -s ompt_pre_init
