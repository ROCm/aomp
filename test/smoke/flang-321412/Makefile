include ../../Makefile.defs

TESTNAME     = flang-321412
TESTSRC_MAIN = flang-321412.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) -g

include ../Makefile.rules
