include ../../Makefile.defs

TESTNAME  = vectoradd_hip
TESTSRC_MAIN = vectoradd_hip.o
TESTSRC_AUX	= vectoradd_hip2.o

$(TESTRC_MAIN)	: vectoradd_hip.cpp
$(TESTSRC_AUX)	: vectoradd_hip2.cpp

TESTSRC_ALL	= $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS	= -fgpu-rdc
CLANG	= hipcc
OMP_BIN	= $(AOMP)/bin/$(CLANG)
CC	= $(OMP_BIN)

include ../Makefile.rules
