include ../../Makefile.defs

TESTNAME     = associate_ptr
TESTSRC_MAIN = associate_ptr.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++ -DUSE_HIP -D__HIP_PLATFORM_AMD__ -I${AOMPHIP}/include -L${AOMPHIP}/lib -lamdhip64 -g
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
