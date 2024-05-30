include ../../Makefile.defs

# This particular test will primarily use a bash script to handle the build via
# cmake and actual execution.
# Hence, we build a dummy target to remain compliant.

TESTNAME     = veccopy-ompt-target-cmake
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
DUMMYSRC     = echo 'int main(void) { return 0; }'

RUNCMD       = ./performBuildRun.sh -a $(AOMP) -t $(AOMP_GPU)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
RUNENV       += OMPTEST_RUN_AS_TESTSUITE=1 CPLUS_INCLUDE_PATH=$(AOMP)/lib64/omptest/include
DUMMYOPT     = $(VERBOSE) -x c -
CC           = $(DUMMYSRC) | $(OMP_BIN) $(DUMMYOPT)

include ../Makefile.rules

clean::
	git clean -dfx ./build
