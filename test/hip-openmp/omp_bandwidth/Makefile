include ../../Makefile.defs

MP4BUILD = 1
#host_debug_flag = -g
host_debug_flag ?=

DEBUG_LEVEL ?= 0

ifeq ($(VERBOSE),1)
  $(info    Compilation and linking VERBOSE Mode ON)
  VERBOSE = -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  VERBOSE += -save-temps
endif

ifdef MP4BUILD
CLANG=$(AOMP)/bin/clang++

DSIZE ?= 1000000000

CXXFLAGS = $(host_debug_flag) -O3 -I${OMPIDIR}/include $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) -Xopenmp-target=$(AOMP_GPUTARGET) -march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES) $(VERBOSE) $(CDEFS) -DDSIZE=$(DSIZE)
CXX = $(AOMP)/bin/clang++
LFLAGS = -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) $(AOMP_CPUTARGET)
ifeq (nvptx,$(findstring nvptx,$(AOMP_GPUTARGET)))
   CUDA ?= /usr/local/cuda
   LFLAGS += -L$(CUDA)/lib64 -lcuda -lcudart -lelf -lffi
endif
CXXLNKFLAGS = $(host_debug_flag)

else
CXXFLAGS = $(host_debug_flag) -O3 -I${OMPIDIR/include
CXXLNKFLAGS = $(host_debug_flag)
LFLAGS =
LIBS = -lm
endif

ifeq ($(DEBUG_LEVEL),0)
  SETDEBUGLIB =
  SETDEBUGRUNENV =
else
# if DEBUG_LEVEL != 0, build with the debug libraries which are slow.
  SETDEBUGLIB = LIBRARY_PATH=$(AOMP)/lib-debug
# Once you build with debug libs, you have different levels of debug
# by setting these environment variables.
# LIBOMPTARGET_DEBUG: Host device runtime debug messages
  SETDEBUGRUNENV = LIBOMPTARGET_DEBUG=1
# LIBOMPTARGET_DEVICE_RTL_DEBUG: Device runtime debug messages.
#SETDEBUGRUNENV = LIBOMPTARGET_DEVICE_RTL_DEBUG=-1
# ATMI_DEBUG: Print ATMI debug messages
#SETDEBUGRUNENV = ATMI_DEBUG=1
endif

# NOTE: You do NOT need debug libs to turn kernel tracing on.
# Set LIBOMPTARGET_KERNEL_TRACE anytime to get kernel launch trace.
# Uncomment next line to activate kernel launch trace in this makefile.
#SETDEBUGRUNENV += LIBOMPTARGET_KERNEL_TRACE=1

SRCS = 	omp_band.cpp

OBJS = $(SRCS:%.cc=objs/%.o)
all: objs omp_band

objs:
	mkdir objs

omp_band: objs $(OBJS)
	@echo
	@echo linking omp_band
	@echo
	$(SETDEBUGLIB) $(CLANG) $(CXXFLAGS) $(OBJS) $(CXXLNKFLAGS) -o $@ $(LIBS) $(LFLAGS)

run: omp_band
	$(SETDEBUGRUNENV) ./omp_band

objs/omp_band.o: omp_band.cpp
	$(CLANG) $(CXXFLAGS) -c omp_band.cpp -o $@

objs/%.o: %.c
	$(CLANG) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f omp_band objs/*.o
