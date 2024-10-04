include ../../Makefile.defs

TESTNAME     = is_dev_ptr
TESTSRC_MAIN = is_dev_ptr.cc
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

HIP_FLAGS = -I${AOMPHIP}/include -L${AOMPHIP}/lib -lamdhip64 \
	-D__HIP_PLATFORM_AMD__ --offload-arch=gfx90a

CLANG        ?= clang++ $(HIP_FLAGS)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
