include ../../Makefile.defs

TESTNAME     = flang-target-associate-ptr
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CLANG        ?= clang
OMP_BIN      =
CC           = AOMP=${AOMP} AOMP_GPU=${AOMP_GPU} FLANG=${FLANG} CLANG=${CLANG} ./buildit.sh
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
