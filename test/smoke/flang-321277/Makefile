include ../../Makefile.defs

TESTNAME     = flang-321277
TESTSRC_MAIN = flang-321277.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED = ISVIRT_RUNTIME

include ../Makefile.rules
