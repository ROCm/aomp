NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = array_slice_repro
TESTSRC_MAIN = array_slice_repro.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME}

include ../Makefile.rules
