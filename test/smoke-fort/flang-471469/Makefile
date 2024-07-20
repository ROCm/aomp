include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -gpulibc
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

HIPCC           = ${AOMP}/bin/clang -x hip
FC              = ${AOMP}/bin/flang-new
#HIP_LIB_PATH   ?= ${ROCM_PATH}/hip/lib
#HIP_LIB_PATH   ?= /COD/LATEST/aomp/lib
HIP_LIB_PATH   ?= /opt/rocm/lib

# Compiler flags
HIP_FLAGS       = -O3 --offload-arch=${AOMP_GPU}
F_FLAGS         = -O2 -fopenmp -fopenmp-version=50 --offload-arch=${AOMP_GPU}
LD_FLAGS        = -O3 -L${HIP_LIB_PATH} -lamdhip64 -lstdc++

# To verify HIP runs, use:
#   AMD_LOG_LEVEL=1 ...
#   AMD_LOG_LEVEL=3 ...
test:
	mkdir -p obj
	${HIPCC} ${HIP_FLAGS} -c kernel.cpp -o obj/kernel.o
	${FC} ${F_FLAGS} -c driver.f90 -o obj/driver.o
	${FC} ${F_FLAGS} ${LD_FLAGS} -o test obj/driver.o obj/kernel.o
	rm -rf obj

