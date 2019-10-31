#-----------------------------------------------------------------------
#  Makefile: Demo simple makefile for hip with the AOMP compiler
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

AOMP ?= /usr/lib/aomp
CC   =$(AOMP)/bin/hipcc -save-temps

# --- Demo compile and link in one step, no object code saved
math_modff: math_modff.cpp kernel.cpp
	$(CC) $^ -o $@
run: math_modff
	LD_LIBRARY_PATH=$(AOMP)/lib ./math_modff

# --- Demo compile and link in two steps, object saved
math_modff.o: math_modff.cpp
	$(CC) -c $^
kernel.o: kernel.cpp
	$(CC) -c $^
obin:	math_modff.o  kernel.o
	$(CC)  $^ -o obin
run_obin: obin 
	LD_LIBRARY_PATH=$(AOMP)/lib ./obin

clean:
	@[ -f ./math_modff ] && rm ./math_modff ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./math_modff.o ] && rm math_modff.o ; true
	@[ -f ./kernel.o ] && rm kernel.o ; true
