# NOOPT        = 1
# NOOMP        = 1
# OMP_FLANGS   = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = flang_red_bug_51
TESTSRC_MAIN = flang_red_bug_51.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
