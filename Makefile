#-----------------------------------------------------------------------
#
#  Makefile: Makefile to build aomp from release source code/tarball
#
# Set location of symbolic link AOMP. Link will be to versioned AOMP install in parent of ${AOMP)
AOMP ?= /usr/local/aomp
AOMP_REPOS = $(shell pwd)
all: 
	AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 $(AOMP_REPOS)/aomp/bin/build_aomp.sh
install:
	@echo "Installation complete to $(AOMP)"
