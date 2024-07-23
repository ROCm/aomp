include ../../Makefile.defs

TESTNAME     = flang-462537
TESTSRC_MAIN = flang-462537.f90
TESTSRC_AUX  = constants.f90
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang-new
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules

flang-462537: $(TESTSRC_ALL)
	$(CC) $(OMP_FLAGS) -c constants.f90
	$(CC) $(OMP_FLAGS) -c flang-462537.f90
	ar r constants.a constants.o
	$(AOMP)/bin/$(CLANG) -lm -L$(AOMP)/lib -lFortranRuntime -lFortranDecimal -lomp -lomptarget $(OMP_FLAGS) -o flang-462537 flang-462537.o constants.a
