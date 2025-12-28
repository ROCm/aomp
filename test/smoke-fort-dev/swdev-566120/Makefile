#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = gpu_kernel_latency
TESTSRC_MAIN = gpu_kernel_latency.F90
TESTSRC_AUX  = gpu_kernel_latency_mod.F90
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        = flang
CFLAGS       = $(FLANG_GPU_LINK_FLAGS) -O2 -g -fopenmp
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME} 

include ../Makefile.rules
