include ../../Makefile.defs

TESTNAME     = AsyncEvents
TESTSRC_MAIN = async-events.cpp
TESTSRC_AUX  = empty-sink.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

AOMPROCM     ?= $(AOMPHIP)
ifneq (,$(wildcard $(AOMPROCM)/bin/rocprof))
	RUNPROF   = $(AOMPROCM)/bin/rocprof
else
	RUNPROF   = $(AOMPROCM)/../bin/rocprof
endif

# If 'ROCR_VISIBLE_DEVICES' is empty, default to zero. Otherwise, split on ','
# and use the first device marked visible by the environmental variable. The
# latter will keep a given value if it does not contain ','.
ifeq ($(ROCR_VISIBLE_DEVICES),)
    RUNENV    = ROCR_VISIBLE_DEVICES=0
else
    comma    := ,
    DEVICE    =$(word 1,$(subst $(comma), ,$(ROCR_VISIBLE_DEVICES)))
    RUNENV    = ROCR_VISIBLE_DEVICES=$(DEVICE)
endif

RUNPROF_FLAGS = --hsa-trace

RUNENV       += OMP_NUM_THREADS=2 LIBOMPTARGET_AMDGPU_HSA_QUEUE_BUSY_TRACKING=1
RUNCMD       = ./$(TESTNAME) && python3 countQueueCreateEvents.py 2

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
