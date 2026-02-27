include ../../Makefile.defs

TESTNAME     = clang_udel_saxpy
TESTSRC_MAIN = clang_udel_saxpy.cc
TESTSRC_AUX  = 
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
