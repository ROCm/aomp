include ../../Makefile.defs

TESTNAME     = flang-thread-limit
TESTSRC_MAIN = flang-thread-limit.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -fopenmp-target-new-runtime  -fenable-host-devmem

include ../Makefile.rules
