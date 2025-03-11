include ../../Makefile.defs

TESTNAME     = sp
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
FC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

.PHONY: sp
sp:
	${FC} ${OMP_FLAGS} -c share.fppized.f
	${FC} ${OMP_FLAGS} sp.fppized.f share.fppized.o -o sp
