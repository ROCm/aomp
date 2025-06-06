include ../../Makefile.defs

TESTNAME     = host_targ
TESTSRC_MAIN = host_targ.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET 	     = -fopenmp -fopenmp-targets=$(CLANG_HOST_TARGET) -Xopenmp-target=$(CLANG_HOST_TARGET) --march=x86 -save-temps

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
