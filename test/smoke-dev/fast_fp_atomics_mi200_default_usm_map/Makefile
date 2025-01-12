include ../../Makefile.defs

TESTNAME     = fast_fp_atomics_mi200_default_usm_map
TESTSRC_MAIN = fast_fp_atomics_mi200_default_usm_map.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# Do not use ENV var to control GFX90A coarse grain USM map.
# This test checks for default behavior.
RUNENV       += HSA_XNACK=1
SUPPORTED    = gfx90a

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -fopenmp-offload-mandatory
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
