include ../../Makefile.defs

TESTNAME      = dynamic_module_load
TESTSRC_MAIN := dynamic_module_load.c
TESTSRC_SO   := dynamic_module_load_shared.c
TEST_SO      := dynamic_module_load.so
TESTSRC_ALL  := $(TESTSRC_MAIN)
ARGS         := ./$(TEST_SO)

# We do not get double free detected in tcache 2 with clang-linker-wrapper
# which is used with --no-opaque-offload-linker
# FIXARG       := --no-opaque-offload-linker

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       =
OMP_FLAGS    =
LINK_FLAGS   = -ldl

include ../Makefile.rules

# Indirectly add prerequisite TEST_SO to general TESTNAME recipe via TESTSRC_ALL
$(TESTSRC_ALL) : $(TEST_SO)

$(TEST_SO) : $(TESTSRC_SO)
	$(CC) -fopenmp $(TARGET) $(FIXARG) -fPIC -shared $(TESTSRC_SO) -o $(TEST_SO)
