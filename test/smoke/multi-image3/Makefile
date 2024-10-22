include ../../Makefile.defs

TESTNAME     = multi-image
TESTSRC_MAIN = multi-image.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD	     = ./$(TESTNAME) && strings $(TESTNAME) | grep $(AOMP_GPU)$(AOMP_TARGET_FEATURES) && strings $(TESTNAME) | grep gfx908$(AOMP_TARGET_FEATURES) && strings $(TESTNAME) | grep gfx942$(AOMP_TARGET_FEATURES)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)$(AOMP_TARGET_FEATURES),gfx908$(AOMP_TARGET_FEATURES),gfx942$(AOMP_TARGET_FEATURES)
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
