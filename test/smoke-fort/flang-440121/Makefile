include ../../Makefile.defs

TESTNAME     = flang-440121
TESTSRC_MAIN = flang-440121.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
WRONG_LIB    = `pwd`/lib
CC           = LIBRARY_PATH=$(WRONG_LIB):$(AOMP)/lib $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -fopenmp
RUNENV       = LD_LIBRARY_PATH=$(WRONG_LIB):$(AOMP)/lib:$(AOMP)/lib/x86_64-unknown-linux-gnu

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
