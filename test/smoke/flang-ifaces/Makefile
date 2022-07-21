include ../../Makefile.defs

TESTNAME     = flang-ifaces
TESTSRC_MAIN = flang-ifaces.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
OMP_FLAGS += -O0

include ../Makefile.rules
