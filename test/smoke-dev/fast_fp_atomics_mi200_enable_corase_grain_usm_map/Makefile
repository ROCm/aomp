include ../../Makefile.defs

TESTNAME     = fast_fp_atomics_mi200_enable_coarse_grain_usm_map
TESTSRC_MAIN = fast_fp_atomics_mi200_enable_coarse_grain_usm_map.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)#

# ENV var spelling different depending on the rocm versions. List both.
RUNENV       += HSA_XNACK=1 OMPX_DISABLE_USM_MAPS=0 OMPX_ENABLE_GFX90A_COARSE_GRAIN_USM_MAPS=1
SUPPORTED    = gfx90a

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -fopenmp-offload-mandatory
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
