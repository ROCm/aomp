include ../../Makefile.defs

TESTNAME     = flang-320332
TESTSRC_MAIN = flang-320332.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
CFLAGS       = -DLEN_2 -DENTER_DATA
#OMP_FLAGS    = -DLEN_2 -DTARGET_DATA
#OMP_FLAGS    = -DLEN_32 -DENTER_DATA
#OMP_FLAGS    = -DLEN_32 -DTARGET_DATA
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
