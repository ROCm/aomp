include ../../Makefile.defs

TESTNAME        = liba_bundled
TESTSRC_MAIN    = main.c
TESTSRC_AUX     = MyDeviceLib/libMyDeviceLib.a MyHostLib/libMyHostLib.a hfunc0.o
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   = -lMyDeviceLib -LMyDeviceLib
EXTRA_OMP_FLAGS =

CC        = $(AOMP)/bin/clang
UNAMEP    = $(shell uname -m)
HOST_TARGET = $(UNAMEP)-pc-linux-gnu
EXTRA_CFLAGS = -target $(HOST_TARGET)

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  TRIPLE  = nvptx64-nvidia-cuda
  ARCH    = nvptx
else
  TRIPLE  = amdgcn-amd-amdhsa
  ARCH    = amdgcn
endif

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  GPUTYPE = nvptx
else
  GPUTYPE = amdgcn
endif

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
run ::
	echo "This does not run on nvptx yet"
clean ::
	echo "This does not run on nvptx yet"
else

# Build the host and device libraries
MyDeviceLib/libMyDeviceLib.a : MyDeviceLib/Makefile MyDeviceLib/func_1v.c \
				MyDeviceLib/func_2v.c MyDeviceLib/func_3v.c \
				MyHostLib/Makefile MyHostLib/hfunc1.c \
				MyHostLib/hfunc2.c MyHostLib/hfunc3.c hfunc0.c
	AOMP_GPU=$(AOMP_GPU) make -C MyDeviceLib libMyDeviceLib.a; \
	make -C MyHostLib libMyHostLib.a; \
	$(CC) -c -target $(HOST_TARGET) -fopenmp -fopenmp-targets=$(TRIPLE) \
	-Xopenmp-target=$(TRIPLE) -march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES) -O2 hfunc0.c -o hfunc0.o


clean ::
	make -C MyDeviceLib clean; make -C MyHostLib clean; rm -f hfunc0.o

include ../Makefile.rules
endif
