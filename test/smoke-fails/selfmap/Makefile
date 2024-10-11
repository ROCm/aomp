include ../../Makefile.defs

TESTNAME     = selfmap
TESTSRC_MAIN = selfmap.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD_COPY      = HSA_XNACK=0 ./$(TESTNAME)
RUNCMD_AUTOZERO  = HSA_XNACK=1 ./$(TESTNAME)


CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run: $(TESTNAME)
	$(RUNCMD_COPY)
	$(RUNCMD_AUTOZERO)