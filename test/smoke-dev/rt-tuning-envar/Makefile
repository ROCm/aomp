include ../../Makefile.defs

TESTNAME     = rt_tuning_envar
TESTSRC_MAIN = rt_tuning_envar.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV       += OMPX_ENABLE_RUNTIME_AUTOTUNING=1
RUNENV       += LIBOMPTARGET_KERNEL_TRACE=1

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
