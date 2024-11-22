include ../../Makefile.defs

TESTNAME     = flang-use-device
TESTSRC_MAIN = use-device_fortran.f90 use-device_c.c
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang-new
CLANG        ?= clang
OMP_BIN      ?= $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules

flang-use-device: $(TESTSRC_ALL)
	$(AOMP)/bin/$(CLANG) $(VERBOSE) $(OMP_FLAGS) -c use-device_c.c
	$(AOMP)/bin/$(FLANG) $(VERBOSE) $(OMP_FLAGS) -c use-device_fortran.f90
	$(AOMP)/bin/$(CLANG)  -L$(AOMP)/lib -lFortranRuntime -lFortranDecimal  -lm  $(OMP_FLAGS) -lomptarget use-device_c.o use-device_fortran.o -o ${TESTNAME}
