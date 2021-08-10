include ../../Makefile.defs

TESTNAME     = virtunc1
TESTSRC_MAIN = virtunc1.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++ -D__HIP_PLATFORM_AMD__=1 -L $(AOMP)/../hip/lib -l amdhip64
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
