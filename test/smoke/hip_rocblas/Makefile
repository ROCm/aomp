include ../Makefile.defs
$(warning "This test requires rocblas to be installed in /opt/rocm/rocblas. Rocblas works on AMD only.")

TESTNAME     = hip_rocblas
TESTSRC_MAIN = hip_rocblas.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
OMP_FLAGS    =  --offload-arch=${AOMP_GPU}
AOMPROCM ?= /opt/rocm
CFLAGS       = -x hip -I $(AOMPROCM)/rocblas/include/ --offload-arch=${AOMP_GPU} -std=c++11
AOMPHIP ?= $(AOMP)
VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
ifeq ($(shell expr $(VERS) \>= 12.0), 1)
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib
  LLIB = -L$(AOMPHIP)/lib
endif
LINK_FLAGS =  -L $(AOMPROCM)/rocblas/lib -lrocblas -Wl,-rpath,$(AOMPROCM)/rocblas/lib $(LLIB) -lamdhip64  $(RPTH)

ifeq ($(EPSDB),1)
OMP_BIN      = $(AOMPROCM)/bin/hipcc
else
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
endif
CC           = $(OMP_BIN) $(VERBOSE)

#-ccc-print-phases
#"-\#\#\#"
#$(shell $(WARNING))
include ../Makefile.rules
