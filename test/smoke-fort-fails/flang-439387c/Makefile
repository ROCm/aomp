NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -DNO_OMP -c
include ../../Makefile.defs

TESTNAME     = crayptralldim
TESTSRC_MAIN = crayptralldim.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNCMD       = echo Compile only

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
