# NOOPT        = 1
# NOOMP        = 1
# OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = intdo
TESTSRC_MAIN = intdo.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

include ../Makefile.rules
