include ../../Makefile.defs

TESTNAME     = hip-heap-buffer-overflow
TESTSRC_MAIN = hip-heap-buffer-overflow.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9]+')
ifeq ($(shell expr $(VERS) \>= 12.0), 1)
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib/asan
  LLIB = -L$(AOMPHIP)/lib/asan
endif

CFLAGS = -x hip -std=c++11 $(LLIB) -lamdhip64 $(RPTH)

HSA_XNACK ?= 1
RUNENV      += HSA_XNACK=$(HSA_XNACK)

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) --strict-whitespace --match-full-lines $(TESTSRC_MAIN)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = gfx908:xnack+,gfx90a:xnack+,gfx940:xnack+,gfx941:xnack+,gfx942:xnack+

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
