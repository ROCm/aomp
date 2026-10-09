#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = equiv
TESTSRC_MAIN = equiv.f90
TESTSRC_ALL  = $(TESTSRC_MAIN)

FLANG        ?= amdflang
CFLAGS       = -fopenmp
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./equiv

include ../Makefile.rules
