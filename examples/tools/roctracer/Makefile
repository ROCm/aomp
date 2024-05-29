#-----------------------------------------------------------------------
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn GPU targets begin with "gfx". nvptx targets begin
#            with sm_.  Example: To build and run on k4000 do this:
#
#            export LLVM_GPU_ARCH=sm_30
#            make run
#
#  Run "make help" to see other options for this Makefile

RUNALLSH = run.sh
MAKEALLSH = make_all.sh

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  RUNALLSH := $(mkfile_dir)$(RUNALLSH)
  MAKEALLSH := $(mkfile_dir)$(MAKEALLSH)
else
  MAKEALLSH := ./$(MAKEALLSH)
  RUNALLSH := ./$(RUNALLSH)
endif
#include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# ----- Demo compile and link in one step, no object code saved
/tmp/$(USER)/roctrace_test:
	$(MAKEALLSH)

run: /tmp/$(USER)/roctrace_test
	$(RUNALLSH)

clean:
	rm -rf /tmp/$(USER)/roctrace_test
