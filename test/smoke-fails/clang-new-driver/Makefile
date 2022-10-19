include ../../Makefile.defs

TESTNAME     = clang-new-driver
TESTSRC_MAIN = clang-new-driver.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += --offload-new-driver -foffload-lto
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
