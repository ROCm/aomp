include ../../Makefile.defs

TESTNAME     = hip-log
TESTSRC_MAIN = test_log.hip
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RPTH = -Wl,-rpath,$(AOMPHIP)/lib

CLANG        = hipcc
OMP_FLAGS    = 
OMP_BIN      = $(AOMPHIP)/bin/$(CLANG)
CC           = HIPCC_VERBOSE=1 $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

