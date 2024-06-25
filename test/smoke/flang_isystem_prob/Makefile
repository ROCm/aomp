include ../../Makefile.defs

TESTNAME     = flang_isystem_prob
TESTSRC_MAIN = flang_isystem_prob.f90
TESTSRC_AUX  = mod.f90
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

MPI_ROOT ?= minimpi

TARGET       = -isystem $(MPI_ROOT)/include -isystem $(MPI_ROOT)/lib
OMP_FLAGS    =

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG) $(TARGET)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
$(TESTNAME):
	$(CC) -J minimpi/lib/ mod.f90 -c
	$(CC) -isystem $(MPI_ROOT)/include -isystem $(MPI_ROOT)/lib $(TESTSRC_MAIN) -o $(TESTNAME)

run: $(TESTNAME)
