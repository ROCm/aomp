include ../../Makefile.defs

TESTNAME     = p
TESTSRC_MAIN = p.f90
TESTSRC_AUX  =
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = $(FLANG_GPU_LINK_FLAGS)
OMP_FLAGS   += -fopenmp-version=60

include ../Makefile.rules
