CFLAGS ?= -lhipblas -lrocsolver -lrocblas

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir :=
endif
GPUFLAGS      ?= --offload-arch=$(LLVM_GPU_ARCH)

SOURCES0 := $(shell find $(_in_dir) -name "*.f03" | sed "s,^\./,,g")
SRCNODIR := $(notdir $(SOURCES0))
APPS     := $(SRCNODIR:.f03=)
RUN_APPS := $(addprefix run., $(APPS))

.PHONY: build run clean $(APPS) $(RUN_APPS)

build: $(APPS)

$(APPS): %:
	$(LLVM_INSTALL_DIR)/../../bin/hipfc $(CFLAGS) $(GPUFLAGS) $(_in_dir)$@.f03 -o $@

$(RUN_APPS): run.%: %
	HIP_TRACE_API=1 ./$^

run: $(RUN_APPS) 

clean:
	rm -f $(APPS) *.o *.mod
