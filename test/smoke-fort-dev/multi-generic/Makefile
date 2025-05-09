#NOOPT        = 1
#NOOMP        = 1
#NOLINK       = 1
include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = vadd.f90 vmul.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -c
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

# Note: will only fail on a system with a matching generic architecture
#       vmul is never even called from main, just linked in as an extra object for this test
# good:
# ${OMP_BIN} -c -fopenmp --offload-arch=gfx900,gfx942 vmul.f90
# bad:
# ${OMP_BIN} -c -fopenmp --offload-arch=gfx942,gfx9-4-generic vmul.f90
# good:
# ${OMP_BIN} -c -fopenmp --offload-arch=gfx942,gfx9-generic vmul.f90

GENERIC ?= gfx9-generic,gfx9-4-generic,gfx10-1-generic,gfx10-3-generic,gfx11-generic,gfx12-generic
OFFLOAD = -fopenmp --offload-arch

test:
	${OMP_BIN} -c ${OFFLOAD}=${AOMP_GPU} vadd.f90
	${OMP_BIN} -c ${OFFLOAD}=${AOMP_GPU},${GENERIC} vmul.f90
	${OMP_BIN} -c main.f90
	${OMP_BIN} -o test ${OFFLOAD}=${GENERIC} main.o ${LDFLAGS} vadd.o vmul.o
