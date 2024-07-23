
include ../../Makefile.defs

TESTNAME    = hip-gpu-burn
TESTSRC_ALL = hip-gpu-burn.cpp

ifeq ($(TARGET),)
  TARGET    = --offload-arch=$(AOMP_GPU)
endif

CC          = $(AOMP)/bin/clang++
CFLAGS      = -x hip -O3 $(AOMP_CPUTARGET) -std=c++11

# Add cudart only if we have an Nvidia sm_ target
ifeq (sm_,$(findstring sm_,$(TARGET)))
  LINK_FLAGS += -L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib
  CFLAGS     += -x cuda -I$(CUDA)/include
else
  HIPLIBS = -L $(AOMPHIP)/hip -L $(AOMPHIP)/lib
  VERS    = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
  ifeq ($(shell expr $(VERS) \>= 12.0), 1)
    RPTH  = -Wl,-rpath,$(AOMPHIP)/lib
  endif
  CFLAGS += -x hip $(HIPLIBS) -lamdhip64 $(RPTH)
endif

include ../Makefile.rules

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTSRC_ALL)
	$(CC) -c $(CFLAGS) $(TARGET) $^

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTSRC_ALL)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGET) $^

$(TESTNAME).s: $(TESTSRC_ALL)
	$(CC) -c -S $(CFLAGS) $(TARGET) $^
