include ../../Makefile.defs

TESTNAME     = multi-image
TESTSRC_MAIN = multi-image.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)$(AOMP_TARGET_FEATURES) --offload-arch=gfx803$(AOMP_TARGET_FEATURES) --offload-arch=gfx1031$(AOMP_TARGET_FEATURES)
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	./$(TESTNAME)
	strings $(TESTNAME) |grep -i gfx
