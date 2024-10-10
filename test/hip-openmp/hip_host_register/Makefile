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

include ../../Makefile.defs
AOMP ?= /usr/lib/aomp
DEVICE_LIB_PATH=$(AOMPHIP)/amdgcn/bitcode/

CC   =  DEVICE_LIB_PATH=$(DEVICE_LIB_PATH) $(AOMPHIP)/bin/hipcc -I$(AOMPHIP)/include/hip

hip_host_register: hip_host_register.hip
	$(CC) $^ -o $@ -std=c++11
run: hip_host_register
	LD_LIBRARY_PATH=$(AOMP)/lib ./hip_host_register

clean:
	@[ -f ./hip_host_register ] && rm ./hip_host_register ; true
