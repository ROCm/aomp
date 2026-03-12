NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = teams-reductions
TESTSRC_MAIN = teams-reductions.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = timeout 1m ./${TESTNAME}

include ../Makefile.rules
