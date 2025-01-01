include ../../Makefile.defs

TESTNAME     = clang-506539
TESTSRC_MAIN = clang-506539.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1 ROCR_VISIBLE_DEVICES=" " HSA_XNACK=1

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
