NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = getDevNum
TESTSRC_MAIN = getDevNum.f90
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
