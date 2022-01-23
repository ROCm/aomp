include ../../Makefile.defs

TESTNAME     = clang-317896
TESTSRC_MAIN = clang-317896.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++ -D__HIP_PLATFORM_AMD__   -lamdhip64 -DUSE_HIP_MALLOC -DUSE_POSSIX_MEMALIGN
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
