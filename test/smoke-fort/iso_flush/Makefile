include ../../Makefile.defs

TESTNAME     = iso_flush
TESTSRC_MAIN = iso_flush.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME} && ./chkit.sh ${TESTNAME}

include ../Makefile.rules
