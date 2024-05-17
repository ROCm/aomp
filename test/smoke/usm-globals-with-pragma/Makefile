include ../../Makefile.defs
HSA_XNACK ?= 1
AOMP_TARGET_FEATURES = :xnack+

TESTNAME     = usm_globals_with_pragma
TESTSRC_MAIN = usm_globals_with_pragma.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

SUPPORTED    = $(SUPPORTS_USM)

RUNENV      += HSA_XNACK=$(HSA_XNACK) LIBOMPTARGET_KERNEL_TRACE=2
RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)


#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
