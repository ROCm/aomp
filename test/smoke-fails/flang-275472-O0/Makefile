include ../../Makefile.defs

TESTNAME     = flang-275472-O0
TESTSRC_MAIN = flang-275472-O0.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -O0
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
