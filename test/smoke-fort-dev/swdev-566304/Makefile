#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = rush_larsen_gpu_omp_fort
TESTSRC_MAIN = rush_larsen_gpu_omp_fort.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
CFLAGS       = $(FLANG_GPU_LINK_FLAGS) -O3 -g -fopenmp
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME} 

include ../Makefile.rules
