#-----------------------------------------------------------------------
#
#  Makefile: Makefile to build aomp from release source code/tarball
#
# Set location of symbolic link AOMP. Link will be to versioned AOMP install in parent of ${AOMP)
AOMP ?= $(HOME)/rocm/aomp
AOMP_REPOS = $(shell pwd)
all: 
	AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_APPLY_ROCM_PATCHES=0 TARBALL_INSTALL=1 $(AOMP_REPOS)/aomp/bin/build_prereq.sh
	AOMP_SKIP_FLANG_NEW=1 AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_APPLY_ROCM_PATCHES=0 TARBALL_INSTALL=1 DISABLE_LLVM_TESTS=1 $(AOMP_REPOS)/aomp/bin/build_aomp.sh
install:
	@echo "Installation complete to $(AOMP)"
