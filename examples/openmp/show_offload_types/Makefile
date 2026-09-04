#-----------------------------------------------------------------------
#

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

run: $(show.sh)
	env LLVM_INSTALL_DIR=$(LLVM_INSTALL_DIR) LLVM_GPU_ARCH=$(LLVM_GPU_ARCH) $(mkfile_dir)show-offload-types.sh
