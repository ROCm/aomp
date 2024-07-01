include ../../Makefile.defs

TESTNAME     = flang-339906
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = matrix.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

TESTSRC_AUX_OBJ  = matrix.o
TESTSRC_MAIN_OBJ = main.o
TESTSRC_AR       = matrix.a

include ../Makefile.rules

all: $(TESTNAME)

$(TESTSRC_AUX_OBJ) : $(TESTSRC_AUX)
	$(SETENV) $(CC) -c $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $^ -o $@

$(TESTSRC_MAIN_OBJ) : $(TESTSRC_MAIN)
	$(SETENV) $(CC) -c $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $^ -o $@

$(TESTSRC_AR) : $(TESTSRC_AUX_OBJ)
	ar -r $@ $^

# Add duplicate archive on purpose to see if symbols clash. They should be marked weak to avoid duplication error.
$(TESTNAME) : $(TESTSRC_AUX_OBJ) $(TESTSRC_MAIN_OBJ) $(TESTSRC_AR)
	$(SETENV) $(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $(TESTSRC_AR) $(TESTSRC_AR) $(TESTSRC_MAIN_OBJ) -o $@
