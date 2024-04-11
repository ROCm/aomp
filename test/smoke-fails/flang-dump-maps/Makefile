include ../../Makefile.defs

TESTNAME     = flang-dump-maps
TESTSRC_MAIN = flang-dump-maps.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)


FLANG        = flang -g
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
