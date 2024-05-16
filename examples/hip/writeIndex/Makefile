#-----------------------------------------------------------------------
#  Makefile: Demo simple makefile for hip with the LLVM compiler
#            This makefile uses hipcc. 
#-----------------------------------------------------------------------
# MIT License
# Copyright (c) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
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

TESTSRC1 = writeIndex.cpp
TESTSRC2 = kernel.cpp

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC1 := $(mkfile_dir)$(TESTSRC1)
  TESTSRC2 := $(mkfile_dir)$(TESTSRC2)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir
DEVICE_LIB_PATH=$(HIPDIR)/amdgcn/bitcode/

CC   =  DEVICE_LIB_PATH=$(DEVICE_LIB_PATH) $(HIPCC)

# --- Demo compile and link in one step, no object code saved
writeIndex: $(TESTSRC1) $(TESTSRC2)
	  $(CC) $^ -o $@ -std=c++11
run: writeIndex
	LD_LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib ./writeIndex

# --- Demo compile and link in two steps, object saved
writeIndex.o: $(TESTSRC1)
	$(CC) -c $^
kernel.o: $(TESTSRC2)
	$(CC) -c $^
obin:	writeIndex.o  kernel.o
	$(CC)  $^ -o obin
run_obin: obin 
	LD_LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib ./obin

clean:
	@[ -f ./writeIndex ] && rm ./writeIndex ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./writeIndex.o ] && rm writeIndex.o ; true
	@[ -f ./kernel.o ] && rm kernel.o ; true
