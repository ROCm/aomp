#-----------------------------------------------------------------------
#
#  Makefile: Compile OpenMP test case with ASan Flags.
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

TESTNAME =vecadd-HBO
FILETYPE =cpp

ROCM_LLVM ?= /opt/rocm/llvm

ifeq ($(LLVM_GPU),)
# Default ROCM_GPU is gfx900 which is vega
INSTALLED_GPU  = $(shell $(ROCM_LLVM)/../bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
ROCM_GPU       ?= $(INSTALLED_GPU)
endif

ifeq ($(TARGETS),)
TARGETS =--offload-arch=$(ROCM_GPU):xnack+
endif

CC       =$(ROCM_LLVM)/bin/clang++

VERS = $(shell $(ROCM_LLVM)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
ROCM = $(shell $(ROCM_LLVM)/bin/clang --version | grep -oP 'AMD clang')

ifeq ($(ROCM),)
  ROCMHIP ?= $(ROCM_LLVM)
else
  ROCMHIP ?= $(ROCM_LLVM)/..
endif

ifeq ($(shell expr $(VERS) \>= 12.0), 1)
        RPTH = -Wl,-rpath,$(ROCMHIP)/lib -Wl,-rpath,$(ROCM_LLVM)/lib/clang/$(VERS)/lib/linux
endif

#  compiler automatically adds "libdevice/<target>/ to -L opts
LFLAGS   =-L$(ROCMHIP)/lib $(RPTH)
ASAN_FLAGS = -fsanitize=address -shared-libsan
CFLAGS = -O0 -g -std=c++11 -fopenmp

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(TARGETS) $(ASAN_FLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	HSA_XNACK=1 ./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $(TARGETS) $(ASAN_FLAGS) $^

obin:	$(TESTNAME).o
	$(CC) $(ASAN_FLAGS) $(LFLAGS) $^ -o obin

run_obin: obin
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTNAME).$(FILETYPE)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGETS) $(ASAN_FLAGS) $^

$(TESTNAME).s: $(TESTNAME).$(FILETYPE)
	$(CC)  -c -S $(CFLAGS) $(TARGETS) $(ASAN_FLAGS) $^

help:
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTNAME).$(FILETYPE)"
	@echo "  Compiler: 		$(CC)"
	@echo "  Compiler flags: 	$(CFLAGS)"
	@echo
	@echo "Avalable Targets:"
	@echo "  make			// build binary $(TESTNAME) with ASan flags"
	@echo "  make run		// run $(TESTNAME) with ASan flags"
	@echo "  make $(TESTNAME).o	// compile, be, & assemble : -c"
	@echo "  make obin         	// link step only"
	@echo "  make run_obin     	// run obin "
	@echo "  make $(TESTNAME).s	// compile & backend steps : -c -S"
	@echo "  make $(TESTNAME).ll	// compile step only       : -c -S -emit-llvm"
	@echo "  make clean		// cleanup files"
	@echo "  make help 		// this help"
	@echo
	@echo "Environment Variables:"
	@echo "  ROCM_LLVM  default: /opt/rocm/llvm     value: $(ROCM_LLVM)"
	@echo "  ROCM_GPU   default: $(INSTALLED_GPU)             value: $(ROCM_GPU)"
	@echo "  TARGETS    default: --offload-arch=$(ROCM_GPU)"
	@echo "              value: $(TARGETS)"
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
