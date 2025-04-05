include ../../Makefile.defs

TESTNAME     = rt_kernel_time
TESTSRC_MAIN = rt_kernel_time.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV       += LIBOMPTARGET_KERNEL_EXE_TIME=1
INPUTGEN     = grep -o 'LaunchID:\s*[0-9]\+' | cut -d ':' -f 2 | tr -d ' ' | sort -n | wc -l

RUNCMD       = ./$(TESTNAME) 2>&1 | $(INPUTGEN) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
