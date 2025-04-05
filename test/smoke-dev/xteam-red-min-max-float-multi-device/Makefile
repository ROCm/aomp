include ../../Makefile.defs

TESTNAME     = xteam-red-min-max-float-multi-device
TESTSRC_MAIN = xteam-red-min-max-float-multi-device.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=1 OMPX_APU_MAPS=1 LIBOMPTARGET_NUM_MULTI_DEVICES=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS      += -fopenmp-target-multi-device
EXTRA_LDFLAGS     += -lm
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
