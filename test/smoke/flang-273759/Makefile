include ../../Makefile.defs

TESTNAME     = flang-273759
TESTSRC_MAIN = flang-273759.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run: $(TESTNAME)
	LIBOMPTARGET_KERNEL_TRACE=1 \
	$(RUNENV) ./$(TESTNAME) 1


