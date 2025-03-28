include ../../Makefile.defs

TESTNAME     = square_triangTest
TESTSRC_MAIN = square_triangTest.F90
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang-new
OMP_BIN      ?= $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
