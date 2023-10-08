include ../../Makefile.defs

TESTNAME     = loop_dir_10
TESTSRC_MAIN = loop_dir_10.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1
RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

COMP_OUT  = $(TESTNAME).compile.out
COMP_EXPECTED  = $(TESTNAME).compile.expected

CLANG        ?= clang
CFLAGS  += -mllvm -debug-only=target-teams-loop-codegen
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
$(TESTNAME): $(TESTNAME).c
	$(SETENV) $(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(OMP_FLAGS) $(TESTSRC_MAIN) -o $(TESTNAME) 2>&1 | tee $(COMP_OUT)

compile: $(TESTNAME)
	diff $(COMP_EXPECTED) $(COMP_OUT)

run: compile
	$(RUNENV) $(RUNCMD)

# clean up local files
clean::
	rm -f $(COMP_OUT)
