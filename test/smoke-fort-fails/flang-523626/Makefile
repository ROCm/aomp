# For testing without openmp:
# OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = spread
TESTSRC_MAIN = spread.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       = -lflang_rt.hostdevice
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

#RUNENV      += LIBOMPTARGET_INFO=-1
RUNCMD       = ./doit.sh ${TESTNAME} && ./chkit.sh ${TESTNAME}

include ../Makefile.rules
