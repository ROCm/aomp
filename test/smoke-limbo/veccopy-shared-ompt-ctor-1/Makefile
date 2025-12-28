include ../../Makefile.defs

TESTNAME     = veccopy-ctor-1
TESTSRC_MAIN = veccopy.cpp
TESTSRC_AUX  = veccopy-ctor-1.so
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD      = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
SHARED_CFLAGS += -fPIC -shared
LINK_FLAGS  += -Wl,-rpath,.
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

veccopy-ctor-1.so: veccopy-ctor-1.cpp
	$(OMP_BIN) $(CFLAGS) $(SHARED_CFLAGS) $(OMP_FLAGS) $^ -o $@

