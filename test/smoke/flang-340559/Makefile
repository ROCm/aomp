include ../../Makefile.defs

TESTNAME     = flang-340559
TESTSRC_AUX  = main.f90
TESTSRC_MAIN = matrix.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD       = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -DUSE_GPU -mcode-object-version=5
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
