include ../Makefile.defs

TESTNAME     = hip_host_register
TESTSRC_MAIN = hip_host_register.hip
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
OMP_FLAGS    =  --offload-arch=${AOMP_GPU}
CFLAGS       = -x hip --offload-arch=${AOMP_GPU} -std=c++11 -I$(AOMP)/include/hip
LINK_FLAGS = -lamdhip64

ifeq ($(EPSDB),1)
OMP_BIN      = /opt/rocm/bin/hipcc
else
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
endif
CC           = $(OMP_BIN) $(VERBOSE)

#-ccc-print-phases
#"-\#\#\#"
#$(shell $(WARNING))
include ../Makefile.rules
