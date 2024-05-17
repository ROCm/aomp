include ../../Makefile.defs

TESTNAME     = flang-313435
TESTSRC_MAIN = flang-313435.f
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
FFLAGS       =   -m64 -ffast-math  -mcmodel=medium -fno-backslash \
                 -Wuninitialized  -fdefault-integer-8 -Wuninitialized  -mfpmath=sse  -mtune=native
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(FFLAGS)

include ../Makefile.rules
