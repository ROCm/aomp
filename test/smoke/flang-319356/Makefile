include ../../Makefile.defs

TESTNAME     = flang-319356
TESTSRC_MAIN = flang-319356.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) -g
CFLAGS       = -DWITH_ARG -DWITH_ARR -DWITH_CONST

include ../Makefile.rules
