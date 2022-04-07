
include ../../Makefile.defs

TESTNAME     = gpus
TESTSRC_MAIN = gpus.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
LOCAL_OMP_FLAGS = $(filter-out -march=$(AOMP_GPU), $(OMP_FLAGS))
GFXLIST = $(shell grep GFXLIST= ../../../bin/aomp_common_vars | grep -o "gfx.*" | sed -E "s|\"}||" | sed -E "s|gfx1010||")

all: $(GFXLIST)

$(TESTNAME):
	@echo
	@echo ---Skipping default target in Makefile.defs---
	@echo

$(GFXLIST):
	$(CC) $(LOCAL_OMP_FLAGS) -march=$@ gpus.c -o gpus-$@

run: $(GFXLIST)
	@echo
	@echo ---Test is for compilation only for: $(GFXLIST)---
	@echo
clean::
	rm -rf $(TESTNAME) *gfx*

