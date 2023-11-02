include ../../Makefile.defs

TESTNAME     = flang-315421
TESTSRC_MAIN = ft.f90
TESTSRC_AUX  = ft_data.f90 randdp.f90
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -DFAIL

include ../Makefile.rules
