include ../Makefile.defs

TESTNAME     = lto_teams
TESTSRC_MAIN = lto_teams.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
# ALT_LINKER = <PATH to AOCC ld.lld>
ALT_LINKER = lld

CFLAGS	     = -O3 -flto=thin -fuse-ld=$(ALT_LINKER)
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

