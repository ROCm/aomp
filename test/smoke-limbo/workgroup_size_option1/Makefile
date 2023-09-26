include ../../Makefile.defs

TESTNAME     = workgroup_size_option1
TESTSRC_MAIN = workgroup_size_option1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
EXTRA_CFLAGS += -fopenmp-gpu-threads-per-team=1024
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

# Select check prefixes according to wave size. Currently, AOMP_WSIZE
# only defined for size 32.
ifndef AOMP_WSIZE
  CHECK_PREFIX = --check-prefixes CHECK,CHECK_WAVE64
else
  CHECK_PREFIX = --check-prefixes CHECK,CHECK_WAVE32
endif

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(CHECK_PREFIX) $(TESTSRC_MAIN)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) $(RUNCMD)
