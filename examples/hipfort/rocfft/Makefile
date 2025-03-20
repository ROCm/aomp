CFLAGS ?= -lrocfft

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir := .
endif
GPUFLAGS      ?= --offload-arch=$(LLVM_GPU_ARCH)

build: rocfft

rocfft:  $(mkfile_dir)rocfft.f03
	$(LLVM_INSTALL_DIR)/../../bin/hipfc $(CFLAGS) $(GPUFLAGS) $^ -o $@

run: rocfft
	./rocfft

clean:
	rm -f rocfft *.o *.mod

