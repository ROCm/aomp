include ../../Makefile.defs

# HipDevCov: device-side coverage drained via the in-tree HSA introspection
# pass. main (TESTNAME) is built by the harness with source-based coverage; it
# hipModuleLoad()s mod.co (built+extracted by run_and_check.sh) whose kernel has
# no host shadow and is therefore only reachable by the HSA drain.

TESTNAME     = main
TESTSRC_MAIN = main.hip
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

AOMPHIP     ?= $(AOMP)
ROCM_PATH   ?= $(AOMPHIP)

# Extra budget: run_and_check.sh also compiles mod.hip and extracts mod.co.
TIMEOUT      = 300s

CFLAGS       = -x hip $(TARGET) -fno-gpu-rdc -fprofile-instr-generate -fcoverage-mapping --rocm-path=$(ROCM_PATH)
LINK_FLAGS   = -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

CC           = $(AOMP)/bin/clang $(VERBOSE)

RUNCMD       = ./run_and_check.sh "$(AOMP)" "$(GPU_W_FEATURES)" "$(FILECHECK)" "$(ROCM_PATH)"

include ../Makefile.rules

clean::
	rm -f mod.co builder builder.*.host-* builder.*.hip-amdgcn-amd-amdhsa--* \
	      *.profraw merged.profdata
