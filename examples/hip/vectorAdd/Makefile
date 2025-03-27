#-----------------------------------------------------------------------
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn targets begin with gfx. nvptx targets begin with sm_
#
#  Run "make help" to see how to use this Makefile
#
#-----------------------------------------------------------------------
# MIT License
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

TESTNAME =vectorAdd
FILETYPE =cpp
TESTSRC = $(TESTNAME).$(FILETYPE)

ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

CC := $(LLVM_INSTALL_DIR)/bin/clang++

# Add cudart only if we have an Nvidia sm_ target
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
  CFLAGS +=-x cuda -I$(CUDA)/include --offload-arch=$(LLVM_GPU_ARCH)
  LFLAGS +=-L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib -std=c++11
else
  CFLAGS += -x hip -std=c++11 --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
  LFLAGS += -L$(HIPDIR)/lib -lamdhip64
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRC)
	$(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTSRC)
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@

obin: $(TESTNAME).o
	$(CCENV) $(CC) $(LFLAGS) $^ -o $@

run_obin: obin
	./obin

include $(mkfile_dir)../../inc/help.mk

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
