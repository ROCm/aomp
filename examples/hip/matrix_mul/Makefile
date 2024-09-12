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

TESTNAME =matrixmul
FILETYPE =cpp
TESTSRC = $(TESTNAME).$(FILETYPE)

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

ifeq ($(TARGETS),)
TARGETS =--offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
endif

CC       =$(LLVM_INSTALL_DIR)/bin/clang++

# Add cudart only if we have an Nvidia sm_ target
ifeq (sm_,$(findstring sm_,$(TARGETS)))
  LFLAGS += -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib
  CFLAGS +=-x cuda -I$(CUDA)/include -std=c++11
else
  VERS = $(shell $(LLVM_INSTALL_DIR)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
  ifeq ($(shell expr $(VERS) \>= 12.0), 1)
    RPTH = -Wl,-rpath,$(HIPDIR)/lib
  endif
  HIPLIBS = -L $(HIPDIR)/hip -L $(HIPDIR)/lib $(RPTH)
  CFLAGS += -x hip  $(HIPLIBS) -lamdhip64 -std=c++11 -mcode-object-version=4
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRC)
	$(CC) $(CFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTSRC)
	$(CC) -c $(CFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o 
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTSRC)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGETS) $^

$(TESTNAME).s: $(TESTSRC)
	$(CC)  -c -S $(CFLAGS) $(TARGETS)  $^

help: 
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTSRC)"
	@echo "  Compiler: 		$(CC)"
	@echo "  Compiler flags: 	$(CFLAGS)"
	@echo
	@echo "Avalable Targets:"
	@echo "  make			// build binary $(TESTNAME)"
	@echo "  make run		// run $(TESTNAME)"
	@echo "  make $(TESTNAME).o	// compile, be, & assemble : -c"
	@echo "  make obin         	// link step only"
	@echo "  make run_obin     	// run obin "
	@echo "  make $(TESTNAME).s	// compile & backend steps : -c -S"
	@echo "  make $(TESTNAME).ll	// compile step only       : -c -S -emit-llvm"
	@echo "  make clean		// cleanup files"
	@echo "  make help 		// this help"
	@echo
	@echo "Link Flags:"
	@echo "  Link flags: $(LFLAGS)"
	@echo

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
