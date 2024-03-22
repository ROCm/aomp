include ../../Makefile.defs

TESTNAME     = flang-433943
TESTSRC_MAIN = flang-433943.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG) -g 
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
