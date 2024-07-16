include ../../Makefile.defs

TESTNAME     = flang-339906
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = matrix.f90
#TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
AOMP_NO_PREREQ = 1

# Skip Makefile.defs, because test needs aux object generated before main binary
TESTSRC_AUX_OBJ  = matrix.o

include ../Makefile.rules

$(TESTSRC_AUX_OBJ) : $(TESTSRC_AUX)
	$(SETENV) $(CC) -c $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $^  -o $@

$(TESTNAME) : $(TESTSRC_MAIN) $(TESTSRC_AUX_OBJ)
	$(SETENV) $(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $^ -o $@

