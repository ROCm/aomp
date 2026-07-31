#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = power.o
TESTSRC_MAIN = power.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -c
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME} && ./chkit.sh ${TESTNAME}

include ../Makefile.rules
