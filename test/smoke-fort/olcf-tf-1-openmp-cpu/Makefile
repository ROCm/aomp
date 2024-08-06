NOOPT        = 1
OMP_FLAGS    = -fopenmp
include ../../Makefile.defs
CFLAGS      += -O3

TESTNAME     = Jacobi
TESTSRC_MAIN = Jacobi.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNCMD       = ./$(TESTNAME) 1024 10

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
