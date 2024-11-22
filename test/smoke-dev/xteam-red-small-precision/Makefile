include ../../Makefile.defs

TESTNAME     = xteam_red_small_precision
TESTSRC_MAIN = xteam_red_small_precision.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS       += -fopenmp-target-fast
ifneq ("$(wildcard $(AOMP)/lib/libmlir_float16_utils.so)","")
  CFLAGS += -lmlir_float16_utils
endif
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
