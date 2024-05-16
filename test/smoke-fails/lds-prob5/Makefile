include ../../Makefile.defs

TESTNAME     = lds_prob5
TESTSRC_MAIN = lds_prob5.keep_ll
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS =
CLANG        = opt -amdgpu-replace-lds-use-with-pointer -S
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	cat lds_prob5
