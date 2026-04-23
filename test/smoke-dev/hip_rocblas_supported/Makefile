include ../../Makefile.defs

TESTNAME     = reproducer
TESTSRC_MAIN = reproducer.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP     ?= $(AOMP)
ROCM_PATH   ?= $(AOMPHIP)
HIPCC       ?= $(ROCM_PATH)/bin/hipcc

# RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CXXFLAGS  = -O2 -std=c++17 -Wall -Wextra
CPPFLAGS  = -I$(ROCM_PATH)/include
LDFLAGS   = -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib
LDLIBS    = -lrocblas -lamdhip64

CFLAGS       += $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)
CLANG        = $(HIPCC)
OMP_BIN      = $(HIPCC)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

#.PHONY: reproducer
#reproducer: reproducer.cpp
#	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

include ../Makefile.rules
