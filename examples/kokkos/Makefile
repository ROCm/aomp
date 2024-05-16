# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../inc/find_gpu_and_install_dir.mk
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_prefix := $(mkfile_dir)
endif

kokkos_examples := \
	$(_in_prefix)helloworld

all: $(kokkos_examples)

clean: TARGET=clean
clean: all
run: TARGET=run
run: all

$(kokkos_examples):
	@echo
	@echo "# ========> Compiler example: $@"
ifeq ($(CURDIR)/,$(mkfile_dir))
	$(MAKE) --no-print-directory -C $@ $(TARGET)
else
	$(MAKE) --no-print-directory -f $@/Makefile $(TARGET)
endif

.PHONY: all clean $(kokkos_examples)
