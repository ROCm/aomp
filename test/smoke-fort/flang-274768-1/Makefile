NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = assumed-nosize
TESTSRC_MAIN = assumed-nosize.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = AOMP=${AOMP} AOMP_GPU=${AOMP_GPU} FLANG=${FLANG} ./buildit.sh
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = cat /dev/null

include ../Makefile.rules
