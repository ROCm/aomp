#-----------------------------------------------------------------------
#
#  Makefile: KOKKOS Makefile for amdgcn target.
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = helloworld

UNAMEP = $(shell uname -m)
AOMP_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
  AOMP_CPUTARGET = ppc64le-linux-gnu
endif
# --- Standard Makefile check for AOMP installation ---
ifeq ("$(wildcard $(AOMP))","")
  ifneq ($(AOMP),)
    $(warning AOMP not found at $(AOMP))
  endif
  AOMP = $(HOME)/rocm/aomp
  ifeq ("$(wildcard $(AOMP))","")
    $(warning AOMP not found at $(AOMP))
    AOMP = /usr/lib/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
#
#
# Default AOMP_GPU is gfx900 which is vega
INSTALLED_GPU   = $(shell $(AOMP)/bin/mygpu -d gfx900)
AOMP_GPU       ?= $(INSTALLED_GPU)
AOMPCC          = $(AOMP)/bin/clang++
AOMP_GPUTARGET ?= amdgcn-amd-amdhsa
KOKKOS_INSTALL_PREFIX ?= $(HOME)

ifeq ($(KOKKOS_USE_HIP),1)
  is_hip    = hip
  KOKKOS_ROOT = $(KOKKOS_INSTALL_PREFIX)/kokkos_hip.$(AOMP_GPU)
else
  is_hip    =
  KOKKOS_ROOT = $(KOKKOS_INSTALL_PREFIX)/kokkos_omp.$(AOMP_GPU)
endif

INCLUDES     = -I$(KOKKOS_ROOT)/include
TARGET_FLAGS = -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) -Xopenmp-target=$(AOMP_GPUTARGET) -march=$(AOMP_GPU) 
CPPFLAGS     = $(INCLUDES) -Wall -Wextra -O3 -DNDEBUG  -w $(TARGET_FLAGS) -std=c++17 -DGTEST_HAS_PTHREAD=0 -DGTEST_HAS_TR1_TUPLE=0 -DKOKKOS_DEPENDENCE -DKOKKOS_WORKAROUND_OPENMPTARGET_CLANG
LINKFLAGS    = $(enableopt) -Wall -Wextra -O3 -DNDEBUG -w $(TARGET_FLAGS) 
KOKKOS_LIB   = $(KOKKOS_ROOT)/lib/libkokkoscore.a

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(AOMP)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CPPFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CPPFLAGS += -save-temps 
endif

CPPFLAGS += $(EXTRA_CPPFLAGS)

$(TESTNAME): $(TESTNAME).o $(KOKKOS_LIB)
	$(CCENV) $(AOMPCC) $(LINKFLAGS) -o $@ $(TESTNAME).o $(KOKKOS_LIB) -ldl

$(TESTNAME).o: $(TESTNAME).cpp $(KOKKOS_LIB)
	$(CCENV) $(AOMPCC) $(CPPFLAGS) -o $@ -c $(TESTNAME).cpp 

run: $(TESTNAME) 
	$(RUNENV) ./$(TESTNAME)

$(KOKKOS_LIB):
	$(AOMP)/bin/kokkos_build.sh $(is_hip)

help:
	@echo
	@echo "Source[s]:		$(TESTNAME).cpp"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(AOMP_GPU)"
	@echo "Target triple:		$(AOMP_GPUTARGET)"
	@echo "AOMP compiler: 		$(CC)"
	@echo "Compile flags:		$(CPPFLAGS)"
	@echo "KOKKOS install prefix:	$(KOKKOS_INSTALL_PREFIX)"
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			 // Builds $(TESTNAME) and KOKKOS "
	@echo " make run		 // Executes $(TESTNAME) "
	@echo
	@echo " make clean              // Deletes test build  "
	@echo " make clean_kokkos       // Deletes KOKKOS build "
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  AOMP=<dir>           AOMP install dir, default=/usr/lib/aomp"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
	@echo "  CUDA=<dir>           CUDA install dir, default=/usr/local/cuda"
endif
	@echo

# Cleanup anything this makefile can create
clean:
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin

clean_kokkos:
	rm -rf $(KOKKOS_ROOT)
