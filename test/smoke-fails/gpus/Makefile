include ../../Makefile.defs

TESTNAME     = gpus
TESTSRC_MAIN = gpus.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
LOCAL_OMP_FLAGS_NO_OFFLOAD = $(filter-out --offload-arch=$(AOMP_GPU), $(OMP_FLAGS))
LOCAL_OMP_FLAGS_NO_ARCH = $(filter-out -march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES), $(LOCAL_OMP_FLAGS_NO_OFFLOAD))
LOCAL_OMP_FLAGS_NO_DEFINE = $(filter-out -D__OFFLOAD_ARCH_$(INSTALLED_GPU)__, $(LOCAL_OMP_FLAGS_NO_ARCH))
LOCAL_OMP_FLAGS_NO_TARGET = $(filter-out -fopenmp-targets=amdgcn-amd-amdhsa, $(LOCAL_OMP_FLAGS_NO_DEFINE))
LOCAL_OMP_FLAGS = $(filter-out -Xopenmp-target=amdgcn-amd-amdhsa, $(LOCAL_OMP_FLAGS_NO_TARGET))

# Process GFXLIST:
#   (1) Replace double with single semicolons
#   (2) Replace semicolons with comma
#   (3) Escape ':'
OFFLOAD_GFXLIST := $(shell echo "$(GFXLIST)" | sed 's/;;/;/g' | sed 's/;/,/g' | sed 's/:/\\:/g')

all: $(OFFLOAD_GFXLIST)

$(TESTNAME):
	@echo
	@echo ---Skipping default target in Makefile.defs---
	@echo

$(OFFLOAD_GFXLIST):
	$(CC) $(LOCAL_OMP_FLAGS) --offload-arch=$@$(AOMP_TARGET_FEATURES) gpus.c -o gpus-$@$(AOMP_TARGET_FEATURES)

run: $(OFFLOAD_GFXLIST)
	@echo
	@echo ---Test is for compilation only for: $(OFFLOAD_GFXLIST)---
	@echo

clean::
	rm -rf $(TESTNAME) *gfx*

