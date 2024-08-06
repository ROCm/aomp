include ../../Makefile.defs

TESTNAME     = targ-279975
TESTSRC_MAIN = targ-279975.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++ -I/usr/local/fftw-3.3.8/include -I../dfft -DPENCIL=1 -I../initializer  -I../halo_finder -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER  -I../simulation -DFFTW3=1 -DFFTW3_THREADS=1 -O3 -fopenmp -Wall -I/usr/local/openmpi/include -Werror -fopenmp -std=c++17 -Wno-unused-function -D__HIP_PLATFORM_HCC__ -fopenmp -I/opt/rocm/include -Wno-unused-result -I.
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
