include ../../Makefile.defs

TESTNAME     = mpi-openmp-targ
TESTSRC_MAIN = mpi-openmp-targ.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += ROCR_VISIBLE_DEVICES=0
$(info $(AOMP))

CLANG        = clang
MPICC        = mpicc

OMP_BIN      = OMPI_CC=$(AOMP)/bin/$(CLANG) $(OMPIDIR)/bin/$(MPICC)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	LD_LIBRARY_PATH=$(OMPIDIR)/lib:$(LD_LIBRARY_PATH) $(OMPIDIR)/bin/mpirun -np 2 ./$(TESTNAME)

