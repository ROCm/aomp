include ../Makefile.defs

TESTNAME        = liba_bundled
TESTSRC_MAIN    = main.c
TESTSRC_AUX     = MyDeviceLib/libMyDeviceLib.a
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           = clang
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   =
EXTRA_OMP_FLAGS =

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
MyDeviceLib/libMyDeviceLib.a : MyDeviceLib/Makefile MyDeviceLib/func_1v.c MyDeviceLib/func_2v.c MyDeviceLib/func_3v.c
	AOMP_GPU=$(AOMP_GPU) make -C MyDeviceLib libMyDeviceLib.a

clean ::
	make -C MyDeviceLib clean

include ../Makefile.rules
endif
