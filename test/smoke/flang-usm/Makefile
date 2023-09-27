USE_OFFLOAD_ARCH = 1
AOMP_TARGET_FEATURES=:xnack-

include ../../Makefile.defs

TESTNAME     = flang-usm
TESTSRC_MAIN = flang-usm.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED  = gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1150,gfx1151

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
