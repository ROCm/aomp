include ../../Makefile.defs

TESTNAME    = atomic-contended
TESTSRC_ALL = atomic-contended.cpp

ifeq ($(TARGETS),)
TARGETS     = --offload-arch=$(AOMP_GPU)
endif

CC          = $(AOMP)/bin/clang++
CFLAGS      = -O3 -std=c++11 -fopenmp $(TARGETS) -Wno-vla-cxx-extension

RUNCMD      = ./performBuildRun.sh -a $(AOMP) -c $(CC) -f "$(CFLAGS)"          \
                                   -t $(TESTSRC_ALL)

include ../Makefile.rules

#  ----   Demo compile to LLVMIR
llvmir:
	$(RUNCMD) -l

#  ----   Demo compile to AMDGPU assembly
assembly:
	$(RUNCMD) -m

clean::
	rm -f $(TESTNAME)_dpp    $(TESTNAME)_iter    $(TESTNAME)_none              \
          $(TESTNAME)_dpp.ll $(TESTNAME)_iter.ll $(TESTNAME)_none.ll           \
          $(TESTNAME)_dpp.s  $(TESTNAME)_iter.s  $(TESTNAME)_none.s
