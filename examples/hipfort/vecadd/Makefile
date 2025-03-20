mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk
GPUFLAGS  ?= --offload-arch=$(LLVM_GPU_ARCH)

build: main

main:  $(mkfile_dir)/main.f03 $(mkfile_dir)/hip_implementation.cpp
	$(LLVM_INSTALL_DIR)/../../bin/hipfc $(CFLAGS) $(GPUFLAGS) $^ -o $@

run: main
	./main

clean:
	rm -f main *.o *.mod
