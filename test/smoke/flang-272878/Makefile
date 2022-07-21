include ../../Makefile.defs

TESTNAME     = flang-272878
TESTSRC_MAIN = flang-272878.f90
TESTSRC_AUX  = file1.f90
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
