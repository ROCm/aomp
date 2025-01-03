include ../../Makefile.defs

TESTNAME     = AMDGPUMemoryManager_threshold_env
TESTSRC_MAIN = AMDGPUMemoryManager_threshold_env.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV       += LIBOMPTARGET_DEBUG=1
RUNENV       += OMPX_AMD_MEMORY_MANAGER_THRESHOLD_EXP_2=20

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
