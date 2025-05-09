NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = tgt_firstprivate
TESTSRC_MAIN = test_firstprivate.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN)

FLANG        ?= flang
CFLAGS       = -lflang_rt.hostdevice
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
