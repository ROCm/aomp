include ../../Makefile.defs

TESTNAME     = omp-global-buffer-overflow
TESTSRC_MAIN = omp-global-buffer-overflow.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

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
