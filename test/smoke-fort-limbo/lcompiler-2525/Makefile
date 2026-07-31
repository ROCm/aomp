NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = cggbal.o
TESTSRC_MAIN = cggbal.f
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# -O2 -fPIC -cpp -c cggbal.f -o cggbal.f.o
FLANG        ?= flang
CFLAGS       = -O2 -fPIC -cpp -c
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh > ${TESTNAME}.out 2>&1 && ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
